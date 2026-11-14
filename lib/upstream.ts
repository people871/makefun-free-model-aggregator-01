// lib/upstream.ts — cross-wire translation between the two client wires
// (OpenAI Chat Completions, Anthropic Messages) and the three provider wires
// (OpenAI, Anthropic, Gemini), plus response conversion back to the client wire.
//
// The trial's providers are simulations, but every translation below follows
// the real vendor API of the wire it targets, so the same code runs against
// the real services given real keys and base URLs.
//
// What the translations must preserve (the env's client checks the provider's
// echo of the request against what the client sent, in order):
//   system text, user text, tool names (in order), max_tokens, stop sequences
//   (in order), the image (by the sha8 of its bytes / URL), JSON mode.
// A wire that cannot carry a field (Anthropic has no JSON mode) is never
// chosen for a request that needs it; routing enforces that in main.ts.
//
// Usage is always forwarded verbatim from the provider's answer, never
// recomputed: the client's usage check compares our response's numbers with
// the provider's recorded ones. Likewise the provider's answer text is
// carried whole, and its stop reason is mapped inside its family
// (stop / end_turn / STOP, length / max_tokens, tool_calls / tool_use,
// content_filter / refusal).
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so a replayed trial produces the same bytes.

import { isRecord, type Json } from './providers.ts';

export interface Usage {
  input: number;
  output: number;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

export interface ParsedAnswer {
  text: string;
  /** Stop reason in the provider wire's own spelling. */
  stop: string;
  usage: Usage | null;
  /** Tool calls the provider asked for, if any (never in the simulation). */
  toolCalls?: ToolCall[];
}

export interface ParsedStream {
  text: string;
  stop: string;
  usage: Usage | null;
}

const numOr = (v: unknown, dflt: number): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : dflt;

/** OpenAI finish_reason for a provider stop reason. */
export function toOpenAIStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Anthropic stop_reason for a provider stop reason. */
export function toAnthropicStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a string or of a block array's text blocks, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Split a data: URL into [mediaType, base64] or null. */
function dataUrlParts(url: string): [string, string] | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return [m[1] !== '' ? m[1] : 'image/png', m[2]];
}

/** An OpenAI content part as an Anthropic content block (text / image). */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[0], data: data[1] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. System messages become the system field, user parts become text /
 * image blocks, assistant tool calls become tool_use blocks, tool messages
 * become tool_result blocks; tools, tool_choice, max_tokens, temperature,
 * top_p and stop sequences pass through. JSON mode has no Anthropic
 * spelling, so a body carrying response_format must never be sent here.
 */
export function openAIToAnthropicBody(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        const id = typeof call.id === 'string' ? call.id : '';
        toolNames.set(id, call.function.name);
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: empty input.
          }
        } else if (isRecord(call.function.arguments)) input = call.function.arguments;
        blocks.push({ type: 'tool_use', id, name: call.function.name, input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const content = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content,
          },
        ],
      });
      continue;
    }
    if (role === 'user') {
      const blocks: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const block = openAIPartToAnthropic(part);
          if (block !== null) blocks.push(block);
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI tool definition as a Gemini function declaration. */
function geminiFunctionDeclarations(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    const fn = isRecord(raw.function) ? raw.function : raw;
    if (typeof fn.name !== 'string') return [];
    return [
      {
        name: fn.name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : isRecord(fn.input_schema) ? fn.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** An OpenAI image_url part as a Gemini content part (inlineData / fileData). */
function openAIImageToGemini(url: string): Json | null {
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data[0], data: data[1] } };
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

/** OpenAI messages as Gemini contents; system text is returned separately. */
function openAIContents(body: Json): { systemText: string; contents: Json[] } {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        const id = typeof call.id === 'string' ? call.id : '';
        toolNames.set(id, call.function.name);
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: empty args.
          }
        }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const content = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: toolNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? 'tool', response: { result: content } } }],
      });
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const geminiPart = openAIImageToGemini(part.image_url.url);
            if (geminiPart !== null) parts.push(geminiPart);
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  return { systemText: systemParts.join('\n'), contents };
}

/** Gemini generationConfig for a client request (max tokens, stops, JSON). */
function geminiConfig(opts: { maxTokens: unknown; temperature: unknown; topP: unknown; stops: unknown; json: boolean }): Json {
  const config: Json = {};
  if (typeof opts.maxTokens === 'number') config.maxOutputTokens = opts.maxTokens;
  if (typeof opts.temperature === 'number') config.temperature = opts.temperature;
  if (typeof opts.topP === 'number') config.topP = opts.topP;
  if (typeof opts.stops === 'string') config.stopSequences = [opts.stops];
  else if (Array.isArray(opts.stops)) {
    const stops = opts.stops.filter((s) => typeof s === 'string');
    if (stops.length > 0) config.stopSequences = stops;
  }
  if (opts.json) config.responseMimeType = 'application/json';
  return config;
}

/** OpenAI tool_choice as a Gemini functionCallingConfig mode. */
function geminiCallingMode(choice: unknown): string | null {
  if (choice === 'auto') return 'AUTO';
  if (choice === 'required') return 'ANY';
  if (choice === 'none') return 'NONE';
  if (isRecord(choice) && choice.type === 'function') return 'ANY';
  return null;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGeminiBody(body: Json, _model: string): Json {
  const { systemText, contents } = openAIContents(body);
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiFunctionDeclarations(body.tools);
  if (declarations.length > 0) {
    out.tools = [{ functionDeclarations: declarations }];
    const mode = geminiCallingMode(body.tool_choice);
    if (mode !== null) out.toolConfig = { functionCallingConfig: { mode } };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    temperature: body.temperature,
    topP: body.top_p,
    stops: body.stop,
    json: format === 'json_object' || format === 'json_schema',
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Anthropic messages as Gemini contents; system text is returned separately. */
function anthropicContents(body: Json): { systemText: string; contents: Json[] } {
  const systemText = blocksText(body.system);
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use' && typeof block.name === 'string') {
            if (typeof block.id === 'string') toolNames.set(block.id, block.name);
            parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            if (block.source.type === 'base64' && typeof block.source.data === 'string') {
              parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
            } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
              parts.push({ fileData: { fileUri: block.source.url } });
            }
          } else if (block.type === 'tool_result') {
            const name = toolNames.get(typeof block.tool_use_id === 'string' ? block.tool_use_id : '') ?? 'tool';
            const result = typeof block.content === 'string' ? block.content : blocksText(block.content);
            parts.push({ functionResponse: { name, response: { result } } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  return { systemText, contents };
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGeminiBody(body: Json, _model: string): Json {
  const { systemText, contents } = anthropicContents(body);
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) {
      out.tools = [{ functionDeclarations: declarations }];
      const choice = body.tool_choice;
      const mode = isRecord(choice) ? (choice.type === 'auto' ? 'AUTO' : choice.type === 'any' || choice.type === 'tool' ? 'ANY' : choice.type === 'none' ? 'NONE' : null) : null;
      if (mode !== null) out.toolConfig = { functionCallingConfig: { mode } };
    }
  }
  const config = geminiConfig({ maxTokens: body.max_tokens, temperature: body.temperature, topP: body.top_p, stops: body.stop_sequences, json: false });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* Response parsing: each provider wire's answer -> one ParsedAnswer.  */
/* ------------------------------------------------------------------ */

/** Text of an Anthropic content array (text blocks joined with ""). */
function anthropicAnswerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** An Anthropic Messages JSON answer as a parsed answer. */
export function parseAnthropicJson(body: Json): ParsedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = anthropicAnswerText(content);
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use' || typeof block.name !== 'string') continue;
    toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: block.name, args: isRecord(block.input) ? block.input : {} });
  }
  const usage = isRecord(body.usage)
    ? { input: numOr(body.usage.input_tokens, 0), output: numOr(body.usage.output_tokens, 0) }
    : null;
  return { text, stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn', usage, ...(toolCalls.length > 0 ? { toolCalls } : {}) };
}

/** An OpenAI Chat Completions JSON answer as a parsed answer. */
export function parseOpenAIJson(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : anthropicAnswerText(message.content);
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
    let args: Json = {};
    if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
      try {
        const parsed: unknown = JSON.parse(call.function.arguments);
        if (isRecord(parsed)) args = parsed;
      } catch {
        // Not JSON: empty args.
      }
    } else if (isRecord(call.function.arguments)) args = call.function.arguments;
    toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: call.function.name, args });
  }
  const usage = isRecord(body.usage)
    ? { input: numOr(body.usage.prompt_tokens, 0), output: numOr(body.usage.completion_tokens, 0) }
    : null;
  return { text, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop', usage, ...(toolCalls.length > 0 ? { toolCalls } : {}) };
}

/** Text of a Gemini candidate's parts. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini generateContent JSON answer as a parsed answer. */
export function parseGeminiJson(body: Json): ParsedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiCandidateText(candidate);
  const toolCalls: ToolCall[] = [];
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall) || typeof part.functionCall.name !== 'string') continue;
    toolCalls.push({ id: '', name: part.functionCall.name, args: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const usage = { input: numOr(meta.promptTokenCount, 0), output: numOr(meta.candidatesTokenCount, 0) };
  return { text, stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP', usage, ...(toolCalls.length > 0 ? { toolCalls } : {}) };
}

/** Split SSE text into frames: [{event, data}] with data lines joined. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream as a parsed stream (text, stop, usage). */
export function parseOpenAIStream(text: string): ParsedStream {
  let out = '';
  let stop = 'stop';
  let usage: Usage | null = null;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: numOr(chunk.usage.prompt_tokens, 0), output: numOr(chunk.usage.completion_tokens, 0) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, stop, usage };
}

/** An Anthropic SSE stream as a parsed stream (text, stop, usage). */
export function parseAnthropicStream(text: string): ParsedStream {
  let out = '';
  let stop = 'end_turn';
  let input = 0;
  let output = 0;
  let hasUsage = false;
  for (const frame of sseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOr(data.message.usage.input_tokens, 0);
      hasUsage = true;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) {
        output = numOr(data.usage.output_tokens, 0);
        hasUsage = true;
      }
    }
  }
  return { text: out, stop, usage: hasUsage ? { input, output } : null };
}

/** A Gemini SSE stream as a parsed stream (text, stop, usage). */
export function parseGeminiStream(text: string): ParsedStream {
  let out = '';
  let stop = 'STOP';
  let usage: Usage | null = null;
  for (const frame of sseFrames(text)) {
    const data = jsonOrNull(frame.data.trim());
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      out += geminiCandidateText(candidate);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = { input: numOr(data.usageMetadata.promptTokenCount, 0), output: numOr(data.usageMetadata.candidatesTokenCount, 0) };
    }
  }
  return { text: out, stop, usage };
}

/* ------------------------------------------------------------------ */
/* Response emitting: a parsed answer -> the client wire's bytes.      */
/* ------------------------------------------------------------------ */

const FIXED_CREATED = 1767225600;

/** Split text into 13-char pieces, as the simulated providers stream it. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A parsed answer as an OpenAI Chat Completions JSON body. */
export function emitOpenAIJson(answer: ParsedAnswer, model: string): string {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls !== undefined && answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id !== '' ? call.id : 'call_unknown',
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  return JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toOpenAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage?.input ?? 0,
      completion_tokens: answer.usage?.output ?? 0,
      total_tokens: (answer.usage?.input ?? 0) + (answer.usage?.output ?? 0),
    },
  });
}

/** A parsed answer as an Anthropic Messages JSON body. */
export function emitAnthropicJson(answer: ParsedAnswer, model: string): string {
  const content: Json[] = [{ type: 'text', text: answer.text }];
  if (answer.toolCalls !== undefined) {
    for (const call of answer.toolCalls) {
      content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: call.args });
    }
  }
  return JSON.stringify({
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toAnthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage?.input ?? 0, output_tokens: answer.usage?.output ?? 0 },
  });
}

/** A parsed stream as an OpenAI SSE stream (role chunk, deltas, finish, usage, [DONE]). */
export function emitOpenAIStream(answer: ParsedStream, model: string): string {
  const finish = toOpenAIStop(answer.stop);
  const usage = answer.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let out = `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
  for (const piece of splitText(answer.text)) out += `data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`;
  out += `data: ${JSON.stringify(chunk({}, finish))}\n\n`;
  out += `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } })}\n\n`;
  out += 'data: [DONE]\n\n';
  return out;
}

/** A parsed stream as an Anthropic SSE stream (start, block, deltas, stop, delta, stop). */
export function emitAnthropicStream(answer: ParsedStream, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(answer.text)) {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: toAnthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: usage.output } });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
