# Gap Analysis: openmap vs TencentDB-Agent-Memory / gbrain

Updated: 2026-06-06

openmap is not a generic agent memory system. It is a personal-assistant memory
layer for maps: the core asset is a user's subjective place graph, learned from
conversation, then used to recall places, avoid bad fits, and calibrate fuzzy
spatial words like "near" per situation.

## Reference Patterns

- [TencentDB-Agent-Memory](https://github.com/Tencent/TencentDB-Agent-Memory):
  layered long-term memory, commonly described as L0 conversation -> L1 atom ->
  L2 scenario -> L3 persona, plus compact Mermaid/symbolic state and raw-trace
  drill-down.
- [gbrain](https://github.com/garrytan/gbrain): a brain-first agent memory loop
  with hybrid retrieval, typed graph links, synthesis with citations/gap notes,
  ingestion/enrichment jobs, and eval/replay workflows.

## Where openmap Is Already Differentiated

- **Map-native memory model**: places are first-class nodes with user->place
  relationships, affect, companions, regions, calibration, and anchors.
- **Conversation-only input**: no live POI dependency. The assistant can hand
  openmap raw turns, and openmap records L0 turns plus structured map memory.
- **Subjective spatial self-model**: `near@date`, `near@work`, walk-time, and
  budget are learned from accepted options instead of hardcoded globally.
- **Assistant-ready recall hooks**: `recallContext()` returns stable persona
  context plus per-turn relevant places.
- **Symbolic graph output**: `graphMermaid()` exposes the personal knowledge
  graph in a compact representation.

## Gaps Fixed In This Pass

- **L0 traceability in eval**: `eval/harness.ts` now ingests with `capture()`, so
  eval verifies raw conversation grounding, not only distilled memory.
- **Per-place offline extraction**: the key-free extractor now scopes sentiment,
  concepts, measures, and named regions to the specific quoted place.
- **LLM extraction stability**: LLM intent frames are merged with lexicon frames
  so explicit map words such as `coffee`, `quiet`, `loud`, and `work` survive
  even when the model focuses on latent intent.
- **Goal-conditioned place memory**: observed place goals are stored as
  place-side affordance tags, enabling queries like "quiet work calls" to recall
  the right place.
- **Negative preference promotion**: disliked noisy places now promote to
  `user avoids loud`; negative events no longer create contradictory
  `user likes loud` beliefs.
- **Complex eval coverage**: dataset probes now cover map-assistant recall,
  negative place memory, avoidance, contradiction absence, `near@work`, raw
  conversation search, assistant context injection, and Mermaid graph output.
- **Citation-grade recall**: `recallContext()` now returns raw L0 source turns
  keyed by place id and includes `source turn#...` citations in the recalled
  places block.
- **L1/L2 scenario layer**: every extracted capture now creates a scenario
  object that groups raw turn ids, place ids, concepts, and intents for one
  map-assistant episode.
- **Contradiction repair**: `repairContradictions()` / CLI / MCP repair old DBs
  where earlier inference produced both `likes X` and `avoids X`.
- **Eval replay snapshots**: `npm run eval:replay` emits a JSON snapshot of
  probe results, recall rankings, beliefs, scenarios, and taste profile for
  regression tracking.
- **Replay diff tooling**: `npm run eval:replay:diff -- before.json after.json`
  compares snapshots across probes, recall tops, beliefs, and scenarios, with
  configurable regression/warning thresholds for CI gating.
- **Richer map constraints**: intent frames, prompts, lexicon extraction, and
  ranking now cover open-late, dietary, walkable, low-noise, low-crowd,
  transit, and drive/parking fit, with eval probes for ambient/travel recall.
- **Scenario synthesis quality**: scenario titles now choose the strongest
  canonical intent instead of the model's first broad label, summaries roll up
  preferred/avoided places, and scenario concepts are normalized to the canonical
  map vocabulary instead of raw LLM labels.
- **Long-horizon scenario rollups**: repeated L2 scenarios now derive
  replayable `routines` such as focus/work-study patterns, separating preferred
  and avoided places while preserving scenario ids, concepts, intents, support,
  and confidence. CLI, MCP, eval probes, and replay snapshots expose this layer.
- **Constraint calibration depth**: accepted place context now learns numeric
  thresholds for `noise`, `crowd`, and `transit_walk` in addition to
  `near`, `walk_time`, and `budget`. Heuristic extraction derives values from
  phrases like quiet/uncrowded/next to the subway, LLM extraction is augmented
  with scoped fallback measures, eval probes verify ambient/travel calibration,
  and ranking can use learned thresholds against numeric raw place attributes.
- **Field replay breadth**: `eval/field-dataset.json` adds a second replay
  suite with London/Paris/NYC/Shanghai/Madrid-style assistant transcripts,
  multilingual turns, travel/date/family/business/study workflows, ambient
  calibration, source grounding, symbolic graph checks, and routine rollups.
  `eval:field`, `eval:field:compare`, and `eval:field:replay` run it through
  the same offline/live LLM and replay-diff machinery.
- **Aggregate replay baselines**: `eval/suites.json` now tracks replay suites
  with source, city, language, style, and tag metadata. `npm run eval:all`,
  `eval:all:compare`, and `eval:all:replay` run the full baseline in isolated
  in-memory DBs and emit aggregate pass/fail/category/coverage summaries for CI
  gates and trend dashboards.

## Remaining Gaps / Next Work

- **Real transcript baselines**: the replay pipeline can now aggregate local
  suites and trend-ready snapshots, but production hardening still depends on
  collecting anonymized real assistant transcripts and adding them to
  `eval/suites.json` as non-synthetic suites.
