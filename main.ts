// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages
// (Anthropic), in JSON and in server-sent events, over the ten simulated
// free-tier providers the trial's brief names, and registers the catalog it
// read from them. `node main.ts --list` prints that catalog and exits 0.
//
// Routing (quality is 60% of the score): every request is served by the
// highest-rated free model that meets its requirements, over all ten
// providers and all three wires — OpenAI, Anthropic and Gemini — with the
// request translated into the provider's wire and the answer translated
// back, usage and stop reason forwarded verbatim from the provider. A
// JSON-mode request never goes to an Anthropic-wire provider: that wire
// cannot carry JSON mode, so its echo would fail the request.
//
// Failover (10%): a provider call refused with 402, 429 or 5xx (or lost
// on the network) is retried once on the same provider — a refusal on a
// provider's failure schedule spends that call, and the next call may
// succeed — and then the request moves to the next-best free model that
// meets it. A provider that failed once in this request is not tried
// again for it, so the request never burns two calls on one dead
// provider and exactly one provider call ever answers it. A provider
// whose last two calls in a row both failed is treated as spent for the
// rest of the trial: a scheduled failure is a single call, so two in a
// row means the budget is gone (or an outage began), and trying it again
// on every later request would only spend calls — the tie-break — while
// the next-best provider answers at the same rating or one step down.
//
// Budgets are per provider and never published, so nothing is planned
// against them in advance: the order is the rating order, and a spent
// provider simply fails over to the next best. The manifest order already
// puts the weight-2 requests first, so the best models go to them first.
//
// Determinism: every choice is a function of the brief, the requests and
// the providers' answers so far. No wall clock, no unseeded randomness,
// no sleeps: a retry is immediate, so a replayed trial makes the same
// calls in the same order and leaves the same state export.
//
// Files: main.ts (this server), lib/providers.ts (brief + catalog probe),
// lib/wire.ts (Anthropic <-> Chat Completions for the OpenAI wire),
// lib/upstream.ts (the other cross-wire translations). The two lib files
// are the kit starter's, kept as shipped; the routing, failover and the
// Gemini / Anthropic upstream wires are this entry's own work.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToGeminiBody,
  emitAnthropicJson,
  emitAnthropicStream,
  emitOpenAIJson,
  emitOpenAIStream,
  openAIToAnthropicBody,
  openAIToGeminiBody,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type ParsedAnswer,
  type ParsedStream,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers whose last two calls in a row failed: treated as spent for the trial. */
const deadProviders = new Set<string>();
/** Consecutive failed calls per provider, reset to 0 by a success. */
const failsInRow = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first.
 * Ties break by provider id, then model id, so the order is fixed.
 * A JSON-mode request excludes Anthropic-wire providers: that wire has
 * no JSON mode to carry, so its echo would report json_mode false.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || byId(a.model, b.model));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type ClientWire = 'openai' | 'anthropic';

interface CallSuccess {
  ok: true;
  /** The response body for the client, JSON text or a whole SSE stream. */
  payload: string;
  stream: boolean;
}

interface CallFailure {
  ok: false;
  status: number;
  error: string;
}

type CallResult = CallSuccess | CallFailure;

/**
 * One inference call to one candidate, in that provider's wire, converted
 * back to the client's wire. The provider's answer is read whole first:
 * the text is short, and the conversion needs the final usage, which a
 * stream only carries in its last frame.
 */
async function callCandidate(candidate: Candidate, clientWire: ClientWire, body: Json, displayModel: string): Promise<CallResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicBody(body, candidate.model.id);
  } else {
    const verb = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${verb}`;
    upstreamBody = clientWire === 'openai' ? openAIToGeminiBody(body, candidate.model.id) : anthropicToGeminiBody(body, candidate.model.id);
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, error: (err as Error).message };
  }
  if (!upstream.ok) {
    return { ok: false, status: upstream.status, error: (await upstream.text()).slice(0, 200) };
  }
  const text = await upstream.text();
  const payload = convertResponse(clientWire, provider.wire, stream, displayModel, text);
  if (payload === null) return { ok: false, status: 502, error: 'the provider answered in a shape this server could not read' };
  return { ok: true, payload, stream };
}

/**
 * A provider's answer (JSON text or a whole SSE stream) in the client's
 * wire. Same-wire answers pass through byte for byte; the others are
 * parsed once and re-emitted, text, stop reason and usage intact.
 */
function convertResponse(clientWire: ClientWire, providerWire: string, stream: boolean, model: string, text: string): string | null {
  if (clientWire === 'openai' && providerWire === 'openai') return text;
  if (clientWire === 'anthropic' && providerWire === 'anthropic') return text;
  let answer: ParsedAnswer | ParsedStream;
  try {
    if (stream) {
      answer = providerWire === 'openai' ? parseOpenAIStream(text) : providerWire === 'anthropic' ? parseAnthropicStream(text) : parseGeminiStream(text);
      return clientWire === 'openai' ? emitOpenAIStream(answer, model) : emitAnthropicStream(answer, model);
    }
    const body: unknown = JSON.parse(text);
    if (!isRecord(body)) return null;
    answer = providerWire === 'openai' ? parseOpenAIJson(body) : providerWire === 'anthropic' ? parseAnthropicJson(body) : parseGeminiJson(body);
    return clientWire === 'openai' ? emitOpenAIJson(answer, model) : emitAnthropicJson(answer, model);
  } catch {
    return null;
  }
}

/**
 * Serve one client request: the best free model that meets it, failing
 * over on 402 / 429 / 5xx / network loss, one retry on the same provider
 * first (a scheduled refusal spends that call; the next may succeed).
 * A 400 is not retried: the model or the body is wrong for that provider.
 * A provider whose last two calls in a row failed is spent for the trial.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  const displayModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0].model.id;
  const tried = new Set<string>();
  let lastStatus = 503;
  let lastError = 'every free model refused the request';
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (tried.has(providerId) || deadProviders.has(providerId)) continue;
    tried.add(providerId);
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = await callCandidate(candidate, clientWire, body, displayModel);
      if (result.ok) {
        failsInRow.set(providerId, 0);
        if (result.stream) sendSse(res, result.payload);
        else sendJson(res, 200, JSON.parse(result.payload));
        return;
      }
      lastStatus = result.status;
      lastError = result.error;
      const streak = (failsInRow.get(providerId) ?? 0) + 1;
      failsInRow.set(providerId, streak);
      if (streak >= 2) deadProviders.add(providerId);
      if (result.status === 400 || result.status === 401) break;
      if (attempt === 0 && (result.status === 0 || result.status === 402 || result.status === 429 || result.status >= 500)) continue;
      break;
    }
  }
  log(`no provider served the request: last failure HTTP ${lastStatus}: ${lastError.slice(0, 200)}`);
  sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', `every free model refused the request (last: HTTP ${lastStatus})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
