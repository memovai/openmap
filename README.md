# openmap

**A map-aware memory layer for AI agents.** The agent's conversation is the only
source of memory — openmap extracts places and their attributes from what's said,
remembers them per-user, learns the user's *taste* and their *spatial vocabulary*,
and answers from that memory. It does **not** fetch map/POI data (that's the host
agent's job); openmap is the memory, not the maps API.

```
   conversation ──▶ extract ──▶ [ per-user memory graph ] ──▶ plan · rerank · recall
   (the only input)   places/         events → beliefs            (taste + intent
                      attrs/intent     + calibrations               + learned "near")
```

## What it does

- **Remembers places from conversation** — `capture(turns)` / `observe(turns)` pull
  place mentions + relationship (loved/visited/want…) and **reconcile** them
  (ADD / UPDATE / NOOP) so "want to go to X" → later "went to X, loved it" updates
  one memory instead of duplicating.
- **Infers preferences from behavior** — "searched coffee a few times" → `ask("do I
  like coffee?")` → likely, with provenance. Consolidates into a per-user
  **knowledge graph** of beliefs (likes/avoids/lives_near/pursues…).
- **Resolves latent intent** — a maps query is rarely literal; `recall` turns "a
  cozy date spot" into a frame {goals, companions, vibe, constraints} and ranks the
  user's remembered places by intent × taste × affordances × proximity.
- **Personalizes live place search** — OpenMap does not fetch POIs, but it does
  produce memory-informed search hints before the host calls a map API, then
  reranks the returned candidates by user taste, constraints, prior likes/dislikes,
  and learned spatial vocabulary.
- **Learns the user's spatial vocabulary** — what *near* means to them (e.g. 3 km),
  walk tolerance, typical spend — learned from accepted options, not hardcoded.
- **Map-aware** — home/work anchors, frequented areas (user↔area), distance-aware
  ranking using the *learned* near-radius.

## Install

```bash
npm install        # deps; optional openai + MCP SDK come along
npm run build      # compile to dist/
npm link           # optional: put `openmap` on PATH
```

Public entry points require a model. Set `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`GOOGLE_API_KEY`, or inject the host agent's model with `buildOpenMap(cfg, { llm })`.
If no model is configured, CLI/MCP/library builders fail fast instead of running
a key-free offline heuristic. The heuristic classes still exist for explicit
tests and eval tooling, but they are not a production fallback.

## CLI (JSON to stdout, for agents)

```bash
# ── agent hooks: the two calls a host wires into its turn loop ──
openmap -u alice context "a quiet spot to work"     # before answer → {system, prepend}
openmap -u alice observe transcript.json           # after exchange → raw L0 + extracted memory

# ── live map-search assist: host fetches POIs, openmap personalizes the flow ──
openmap -u alice plan "coffee near me" --near 37.77,-122.42
openmap -u alice rerank "coffee near me" candidates.json --near 37.77,-122.42

# human/debug reads
openmap -u alice search "a cozy quiet spot to work" # ranked remembered places
openmap -u alice evidence "loved the ramen place"  # raw source turns for grounding
openmap -u alice debug graph --mermaid             # inspect derived graph
openmap -u alice debug profile                     # taste profile + stats

# manual overrides are intentionally explicit and off the happy path
openmap -u alice manual remember 'loved "Blue Bottle Coffee"' --relationship loved
openmap -u alice manual calibrate near 3
openmap -u alice manual memory list

openmap serve
```

First-run integration check:

```bash
openmap onboard          # config, privacy notes, and recommended host wiring
openmap onboard --demo   # requires a model; no-write in-memory demo
```

### Auto-learning from conversation

`observe` doesn't ask the agent to fill fields like `--relationship loved`; it
logs the exchange, extracts those relationships from the user's wording, and
learns. When the agent offers options with
distances, prices, ambient fit, or transit access and the user accepts one,
openmap updates the calibration layer automatically (revealed preference):

```
assistant: "Ritual Coffee is 3km away, ¥45 pour-over, quiet, uncrowded, next to the subway"
user:      "let's do Ritual, loved it"
            → remembers Ritual (loved)
            → learns near≈3km, budget≈¥45, noise≈0.2, crowd≈0.2, transit_walk≈2min
```

### Agent integration — auto-recall / auto-capture

Two hooks are all a host agent needs. **auto-recall** before answering, **auto-capture** after:

```ts
// before the agent answers — inject what we remember
const { system, prepend } = await mem.recallContext(userMessage, { userId });
//   system  → stable persona/geography block, cache on the system prompt
//   prepend → relevant remembered places for THIS turn, prepend to the user message

// after the exchange — persist raw turns (for grounding) + distil memory
await mem.capture([{ role: "user", content: userMessage }, { role: "assistant", content: reply }], { userId });
```

For live local search, use the memory layer around the host's map provider:

```ts
// before calling OpenStreetMap/Google/etc.
const plan = await mem.planPlaceSearch("coffee near me", { userId, near });
// plan.searchQuery / plan.include / plan.avoid tell the host what to search for.

const candidates = await hostMapSearch(plan.searchQuery, { near: plan.location });
const ranked = await mem.rankCandidatePlaces("coffee near me", candidates, { userId, near });
// ranked.results are live POIs, personalized but not persisted.
```

When the host map provider returns stable place ids, pass them as `sourceId`
with `source` so future memories canonicalize to the same place instead of
depending on name matching.

Raw turns are kept in an **L0 log** so the agent can recall original wording to ground a
memory — `mem.searchConversation("the loud bar")` / the `conversation_search` MCP tool.
`recallContext()` also returns `sources[placeId]` and includes `source turn#...` citations
in the recalled-places block when a raw turn supports a recalled place. Each capture with
extraction also creates an L2 `scenario` summary grouping turn ids, place ids, concepts,
and intents; repeated scenarios are rolled up on demand as `routines`, such as a durable
"focus: quiet + near transit" pattern across work/study episodes. As a
**Claude Code** hook: run `openmap context "$PROMPT"` on `UserPromptSubmit` to inject
context, and `openmap observe transcript.json` on `Stop` to capture the turn.

## As an MCP server

```bash
npm install @modelcontextprotocol/sdk
openmap serve
```

Primary tools: `local_search_context` (pre-search hints), `rerank_places`
(post-search candidate ranking), `recall_context` (auto-recall), `capture`
(auto-capture), `conversation_search` (raw evidence), `taste_profile`,
`scenarios`, `routines`.

Advanced/manual tools are also exposed for tests and admin flows: `remember`,
`observe`, `recall`, `resolve_intent`, `ask`, `consolidate`, `repair_contradictions`,
`beliefs`, `graph`, `get_persona`, `set_persona`, `set_place_role`,
`add_place_alias`, `place_aliases`, `anchors`, `regions`, `calibrate`,
`calibrations`, `learn_near`, `list_memories`, `forget`, `list_collections`,
`add_to_collection` (all accept an optional `userId`).

## As a library

```ts
import { buildOpenMap } from "openmap";

const mem = buildOpenMap();
await mem.observe([{ role: "user", content: 'dinner at "Quiet Garden Bistro" was perfect' }], { userId: "alice" });
for (const r of await mem.recall("somewhere romantic for dinner", null, 5, "alice")) {
  console.log(r.place.name, r.score, r.reasons);
}
```

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `OPENMAP_DB` | `~/.openmap/openmap.db` | SQLite path |
| `OPENMAP_EMBEDDER` | `auto` | `auto` \| `openai` \| `none` (no key → keyword-only recall) |
| `OPENMAP_TAGGER` | `auto` | `auto` \| `llm`; `lexicon` is disabled for public builders |
| `OPENMAP_MODEL_TAGGER` / `_EXTRACTOR` / `_PERSONA` | base chat model | per-component LLM model |
| `OPENMAP_BELIEF_HALFLIFE_DAYS` | `60` | recency decay for inferred beliefs (0 = off) |
| `OPENAI_API_KEY` | — | enables OpenAI embeddings + LLM extraction |
| `OPENMAP_OPENAI_BASE_URL` | — | **BYOC**: OpenAI-compatible endpoint (DeepSeek / local / gateway) |

### Bring your own model — three ways

LLM extraction (intent frame, mentions, structured memory) runs through an
injectable `LLMRunner`, so you can:

1. **Borrow the host agent's model** — inject a runner; openmap doesn't need its
   own key: `buildOpenMap(cfg, { llm: myRunner })`.
2. **BYOC endpoint** — set `OPENAI_API_KEY` + `OPENMAP_OPENAI_BASE_URL` to any
   OpenAI-compatible backend (DeepSeek, a local server, a Claude gateway).
3. **No offline fallback** — no key and no injected runner → user-facing error.

Each component (`tagger` / `extractor` / `persona`) can use a different model.

Storage is **SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)** (vec0 KNN,
with a brute-force fallback). There is no POI/map-data source to configure.

## Architecture

A thin `OpenMap` facade orchestrates focused modules — see
[`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`MEMORY_MODEL.md`](./MEMORY_MODEL.md):

```
core/    types · geo · config            store/   sqlite + sqlite-vec + migrations + aliases
nlp/     embedding · extract · tagger     search/  planning · candidate rerank · recall · ranking
prompts/ intent · mentions · memory       world/   affordance(vibe) · relations(near/similar)
memory/  inference(beliefs+reconcile+decay) · taste · anchors · regions
         · calibration(near/walk/noise/crowd/transit thresholds) · graph · persona
         · scenarios/routines · hooks(auto-recall/capture)
```

Two linked geo layers: **objective facts** (`core/geo` distances) vs the
**subjective spatial self-model** (`memory/calibration` learned near-radius,
noise/crowd/transit-access thresholds, `memory/regions` frequented areas), joined at query time.

## Development

```bash
npm run typecheck   # tsc --noEmit
npm test            # node:test, network-free via injected fake runners/components
npm run eval        # dataset eval; uses .env.local LLM when configured
npm run eval:compare # offline vs LLM/stub extraction comparison
npm run eval:replay # JSON replay snapshot for ranking/memory regression tracking
npm run eval:all    # aggregate all replay suites from eval/suites.json
npm run eval:all:compare # aggregate offline vs LLM/stub comparison across suites
npm run eval:all:replay -- --out=baseline.json # aggregate replay snapshot for CI/trends
npm run eval:field  # broader field-style replay: cities/languages/use-cases
npm run eval:field:compare # offline vs LLM on the field replay suite
npm run eval:field:replay -- --out=field.json # field replay snapshot
npm run eval -- --dataset=realistic-llm-dataset.json # LLM-only realistic extraction suite
npm run eval:replay:diff -- before.json after.json # compare snapshots; exits non-zero on regressions
npm run eval:providers -- --providers=openmap,mem0,tencentdb,gbrain --dataset=field-dataset.json
npm run build
```

In eval script names, `offline` means the explicit test/eval heuristic baseline
enabled with an internal opt-in. CLI, MCP, and normal `buildOpenMap()` usage still
require a model and report an error when none is configured.

`realistic-llm-dataset.json` is intentionally not part of the heuristic baseline:
it is a realistic conversation-shape extraction suite where every probe is
`llmOnly`. Use it with `.env.local` configured for Gemini/OpenAI-compatible
models to evaluate actual place extraction, alias/correction handling, scoped
sentiment, false positives, and bilingual turns. Regex/lexicon test helpers
should skip this suite rather than define product quality.

### Cross-provider eval

`eval:providers` runs the same dataset/probes against multiple memory systems and
normalizes results into one report. `openmap` runs in-process. `mem0` is optional:
install `mem0ai` and provide an OpenAI-compatible key. `tencentdb` and `gbrain`
are command adapters so we can compare real deployments without vendoring their
runtimes. All providers use the same `.env.local` / `.env` model settings loaded
through `loadConfig()`:

```bash
# shared across openmap, mem0, TencentDB adapters, and gbrain adapters
GEMINI_API_KEY=...
OPENMAP_OPENAI_CHAT_MODEL=gemini-2.5-flash-lite
OPENMAP_OPENAI_EMBED_MODEL=gemini-embedding-001
OPENMAP_OPENAI_EMBED_DIMS=768

# default: openmap, mem0, tencentdb, gbrain
npm run eval:providers -- --dataset=dataset.json --out=provider-report.json

# force deterministic openmap path
npm run eval:providers -- --providers=openmap --offline

# external adapters receive {schema, provider, dataset} on stdin and return
# a ProviderEvalReport-compatible JSON object on stdout. They also inherit
# OPENAI_API_KEY, OPENMAP_OPENAI_BASE_URL, OPENMAP_OPENAI_CHAT_MODEL,
# OPENMAP_OPENAI_EMBED_MODEL, and OPENMAP_OPENAI_EMBED_DIMS.
OPENMAP_EVAL_TENCENTDB_COMMAND="node ./adapters/tencentdb-eval.mjs" \
OPENMAP_EVAL_GBRAIN_COMMAND="gbrain-eval-adapter --json" \
npm run eval:providers -- --providers=openmap,tencentdb,gbrain
```

Unsupported capabilities are counted separately from failures. That keeps the
comparison fair: generic memory systems can score on shared recall behavior
without being penalized for openmap-specific map calibration, aliases, or
place-graph probes they do not implement.

## References

Prior art that informed the memory design (studied, not vendored):
[Mem0](https://github.com/mem0ai/mem0) (extract→update, graph memory),
[TencentDB-Agent-Memory](https://github.com/Tencent/TencentDB-Agent-Memory)
(tiered L0→L3 pyramid, symbolic/Mermaid memory),
[memU](https://github.com/NevaMind-AI/memU) (typed memory categories,
multi-stage retrieval), and gbrain.

## License

MIT.
