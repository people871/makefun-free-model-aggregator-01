# Free Model Aggregator

One server that serves an OpenAI- and an Anthropic-compatible API over ten
simulated free-tier model providers, routing each request to the best free
model that can serve it and failing over when a provider refuses.

Built for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01).

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only. No dependencies to install, no keys in the
repository: provider keys come from the trial brief at run time.

## How it works

- **Catalog discovery.** At startup the server reads the brief
  (`$MAKEFUN_PROMPT_FILE`: provider directory, quality table, workload
  manifest) and probes each provider's own catalog route in its own wire
  (OpenAI `GET /v1/models`, Anthropic `GET /v1/models`, Gemini
  `GET /v1beta/models`). The probed catalog is the document `--list`
  prints and the register body carries, byte for byte.
- **Routing.** For each request, every free model meeting its requirements
  (tools, vision, JSON mode, minimum context) is ranked by the quality
  table's rating, best first, over all ten providers — not just the ones
  whose wire matches the client's. A JSON-mode request never goes to an
  Anthropic-wire provider, because that wire cannot carry JSON mode.
- **Wires.** The request is translated into the provider's wire and the
  answer back into the client's: OpenAI Chat Completions, Anthropic
  Messages and Gemini `generateContent` / `streamGenerateContent`, in JSON
  and in server-sent events. The provider's answer text, usage counts and
  stop reason are forwarded verbatim; the translation preserves the
  fields the provider echoes (system prompt, user text, tool names in
  order, `max_tokens`, stop sequences, images, JSON mode).
- **Failover.** A call refused with 402, 429 or 5xx (or lost on the
  network) is retried once on the same provider — a refusal on a
  provider's failure schedule spends that call and the next may succeed —
  then the request moves to the next-best free model. A provider whose
  last two calls in a row failed is treated as spent for the rest of the
  trial, so later requests stop burning calls on it. Exactly one provider
  call ever answers a request.
- **Determinism.** Every choice is a function of the brief, the requests
  and the providers' answers so far: no wall clock, no unseeded
  randomness, no sleeps. A replayed trial makes the same calls in the
  same order.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll, routing
  and failover.
- `lib/providers.ts` — the brief's types and parser, auth headers, the
  catalog probe (from the kit starter, kept as shipped).
- `lib/wire.ts` — Anthropic <-> Chat Completions translation for the
  OpenAI wire (from the kit starter, kept as shipped).
- `lib/upstream.ts` — the other cross-wire translations: OpenAI and
  Anthropic requests to Gemini and to Anthropic providers, and every
  provider wire's answer back to either client wire.

## Local trial results

Graded locally with the kit (`grade-local --driver process`) on all three
practice instances:

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 28    |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 30    |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 31    |

All 20 requests valid on every trial; every replay matched byte for
byte. Quality 99 is the ceiling on these instances: the only 1475-rated
free model is one provider's, whose per-trial budget (10–12 calls) cannot
cover all 17 non-vision requests, so the remainder are served at 1470.

## License

MIT — see `LICENSE`.
