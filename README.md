# openmap

**A map-aware memory layer for AI agents.** The agent's conversation is the only
source of memory — openmap extracts places and their attributes from what's said,
remembers them per-user, learns the user's *taste* and their *spatial vocabulary*,
and answers from that memory. It does **not** fetch map/POI data (that's the host
agent's job); openmap is the memory, not the maps API.

```
   conversation ──▶ extract ──▶ [ per-user memory graph ] ──▶ recall · ask · persona
   (the only input)   places/         events → beliefs            (taste + intent
                      attrs/intent     + calibrations               + learned "near")
```

## What it does

- **Remembers places from conversation** — `remember(text)` / `observe(turns)` pull
  place mentions + relationship (loved/visited/want…) and **reconcile** them
  (ADD / UPDATE / NOOP) so "want to go to X" → later "went to X, loved it" updates
  one memory instead of duplicating.
- **Infers preferences from behavior** — "searched coffee a few times" → `ask("do I
  like coffee?")` → likely, with provenance. Consolidates into a per-user
  **knowledge graph** of beliefs (likes/avoids/lives_near/pursues…).
- **Resolves latent intent** — a maps query is rarely literal; `recall` turns "a
  cozy date spot" into a frame {goals, companions, vibe, constraints} and ranks the
  user's remembered places by intent × taste × affordances × proximity.
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

Runs offline with **zero keys** (local hashing embedder + lexicon fallback for
extraction). Set `OPENAI_API_KEY` to upgrade embeddings + LLM extraction/intent.

## CLI (JSON to stdout, for agents)

```bash
# ── agent hooks: the two calls a host wires into its turn loop ──
openmap -u alice recall-context "a quiet spot to work"   # auto-recall → {system, prepend} to inject
openmap -u alice capture conversation.json               # auto-capture → log raw turns (L0) + extract
openmap -u alice conversation "loved the ramen place"    # search raw history to ground a memory

# capture from conversation (the only source)
openmap -u alice observe conversation.json        # [{role,content}, …] → extract + reconcile (no L0 log)
openmap -u alice remember 'loved "Blue Bottle Coffee"' --relationship loved

# recall your places by resolved intent
openmap -u alice recall "a cozy quiet spot to work"   # → frame + taste/vibe-ranked places
openmap -u alice intent "romantic dinner with my parents"   # just the resolved frame

# inference + the knowledge graph
openmap -u alice ask "do I like coffee?"          # infer from behavior, with provenance
openmap -u alice consolidate                      # promote events → beliefs
openmap -u alice beliefs                           # the semantic graph edges
openmap -u alice graph --mermaid                   # the knowledge graph as Mermaid

# learned spatial vocabulary + areas
openmap -u alice calibrate near 3                  # "near" ≈ 3km for me (or learn-near 3)
openmap -u alice calibrations                      # near/walk_time/budget/noise/crowd/transit_walk
openmap -u alice anchors                           # home/work/usual area/near radius
openmap -u alice regions                           # areas I'm active in
openmap -u alice places alias "the plant cafe" <placeId> # canonicalize future mentions

# persona, management, MCP
openmap -u alice persona set --likes cozy,wine --dislikes loud
openmap -u alice memory list ; openmap -u alice places list
openmap serve-mcp
```

### Auto-learning from conversation

`observe` doesn't just store — it learns. When the agent offers options with
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

Raw turns are kept in an **L0 log** so the agent can recall original wording to ground a
memory — `mem.searchConversation("the loud bar")` / the `conversation_search` MCP tool.
`recallContext()` also returns `sources[placeId]` and includes `source turn#...` citations
in the recalled-places block when a raw turn supports a recalled place. Each capture with
extraction also creates an L2 `scenario` summary grouping turn ids, place ids, concepts,
and intents; repeated scenarios are rolled up on demand as `routines`, such as a durable
"focus: quiet + near transit" pattern across work/study episodes. As a
**Claude Code** hook: run `openmap recall-context "$PROMPT"` on `UserPromptSubmit` to inject
context, and `openmap capture transcript.json` on `Stop` to capture the turn.

## As an MCP server

```bash
npm install @modelcontextprotocol/sdk
openmap serve-mcp
```

Tools: `recall_context` (auto-recall), `capture` (auto-capture), `conversation_search`,
`scenarios`, `routines`,
`remember`, `observe`, `recall`, `resolve_intent`, `ask`, `consolidate`,
`repair_contradictions`, `beliefs`, `graph`, `taste_profile`, `get_persona`, `set_persona`, `set_place_role`,
`add_place_alias`, `place_aliases`,
`anchors`, `regions`, `calibrate`, `calibrations`, `learn_near`, `list_memories`,
`forget`, `list_collections`, `add_to_collection` (all accept an optional `userId`).

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
| `OPENMAP_TAGGER` | `auto` | `auto` \| `llm` \| `lexicon` (intent/concept extraction) |
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
3. **Offline** — no key, no runner → deterministic lexicon/heuristic fallback.

Each component (`tagger` / `extractor` / `persona`) can use a different model.

Storage is **SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)** (vec0 KNN,
with a brute-force fallback). There is no POI/map-data source to configure.

## Architecture

A thin `OpenMap` facade orchestrates focused modules — see
[`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`MEMORY_MODEL.md`](./MEMORY_MODEL.md):

```
core/    types · geo · config            store/   sqlite + sqlite-vec + migrations + aliases
nlp/     embedding · extract · tagger     search/  recall pipeline · ranking
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
npm test            # node:test, offline & network-free
npm run eval        # dataset eval; uses .env.local LLM when configured
npm run eval:compare # offline vs LLM/stub extraction comparison
npm run eval:replay # JSON replay snapshot for ranking/memory regression tracking
npm run eval:all    # aggregate all replay suites from eval/suites.json
npm run eval:all:compare # aggregate offline vs LLM/stub comparison across suites
npm run eval:all:replay -- --out=baseline.json # aggregate replay snapshot for CI/trends
npm run eval:field  # broader field-style replay: cities/languages/use-cases
npm run eval:field:compare # offline vs LLM on the field replay suite
npm run eval:field:replay -- --out=field.json # field replay snapshot
npm run eval:replay:diff -- before.json after.json # compare snapshots; exits non-zero on regressions
npm run eval:providers -- --providers=openmap,mem0,tencentdb,gbrain --dataset=field-dataset.json
npm run build
```

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
