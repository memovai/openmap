# openmap — Architecture

Full-stack TypeScript (Node ≥ 22.5, ESM). A **map-aware memory layer**: the
agent's conversation is the only input; there is no POI/map-data acquisition.

## Thesis

A personal, geo-anchored **taste + spatial** memory, built entirely from
conversation. `recall` resolves a (usually non-literal) maps query into an intent
frame and ranks the user's *remembered* places by intent × taste × affordances ×
proximity. Two things make it map-aware and personal:

1. a **knowledge graph** of beliefs distilled from behavior (likes/avoids/
   lives_near/pursues), and
2. a **spatial self-model** learned per user — what *near* means to them, where
   they're active — kept separate from objective geo facts and linked at query time.

## Layers (a thin facade over focused modules)

```
┌ interface ── cli.ts · mcp.ts  (JSON-first, agent-consumable)
├ facade ───── openmap.ts  (OpenMap: orchestrates; logic lives below)
├ search ───── ranking.ts  (rankMemory: intent×taste×vibe×geo, dislike penalty)
├ memory ───── inference (beliefs + reconcile ADD/UPDATE/NOOP + recency decay)
│              taste · persona · anchors · regions · calibration · graph
├ nlp ──────── embedding · extract (mentions/concepts/intent/measures) · tagger (IntentFrame)
├ world ────── affordance (vibe) · relations (near/similar via vector KNN)
├ store ────── db.ts  (SQLite + sqlite-vec)
└ core ─────── types · geo (objective distance math) · config
```

### Why these choices

- **Conversation-only.** `remember`/`observe` extract place mentions and attributes
  from text; a name alone is evidence (a place called "Blue Bottle Coffee" implies
  coffee, so it's tagged from its name). No external geocoder/POI API.
- **SQLite + sqlite-vec.** Place embeddings live in a `vec0` virtual table (cosine
  KNN), with a brute-force fallback if the extension can't load. `node:sqlite` =
  no native build; an agent can `npm install` and go.
- **LLM-first NLP, lexicon fallback.** Extraction/intent use an LLM when keyed
  (per-component models); a key-free lexicon keeps it offline and deterministic for
  tests, and handles simple negation/contradiction.
- **Learned, not hardcoded.** "near" is a per-user learned radius (the calibration
  layer), not a constant — and the same mechanism generalizes to walk_time, budget,
  noise: add a term, no new logic.

## Data model

```
places(id, name, lat, lng, category, address, source, tags[], embedding, …)   -- shared nodes
memories(id, user_id, place_id, relationship, affect, note, …)                 -- per-user edges
events(id, user_id, kind, text, place_id, concepts[], intents[], created_at)   -- L0 episodic
beliefs(user_id, subject, predicate, object, confidence, support, source, …)   -- L2 semantic graph
calibrations(user_id, term, value, samples)                                    -- learned fuzzy semantics
personas / collections / collection_items
```

`places` are objective nodes; everything else is per-user. Taste is *derived*
(affect-weighted centroid of loved places, blended with the persona). Beliefs are
the relationship layer; calibrations are the spatial/preference self-model.

## Ranking (auditable; every result carries `reasons`)

```
score = ( W_QUERY·sim(frame, place) + W_AFFECT·affect + W_TASTE·sim(taste, place)
          + W_GEO·geoAffinity(place, learned_near_radius) ) · dislikePenalty · vibeBonus
```

`geoAffinity` uses the user's **learned** near-radius (no hardcoded distance).
`recall` resolves the frame (vibe + goals + concepts), defaults an unanchored query
to the user's anchor (home → most-active area), and logs the query as a behavioral
event — so searching is itself learning.

## Two geo layers, linked

- **Objective facts** — `core/geo` (distances), place coords. Computed, shared.
- **Subjective self-model** — `memory/calibration` (learned "near", walk, budget),
  `memory/regions` (frequented areas = user↔area relationship), anchors.
- **Link** — at query time the subjective semantics parameterize how objective geo
  is read ("near me" → *this* user's radius over real distances from their anchor).

## Extension points

| Want to… | Touch |
|----------|-------|
| Add a learnable fuzzy term | `memory/calibration.ts` `TERMS` (one entry) |
| Better extraction (mentions/measures/relationship) | `nlp/extract.ts` / `nlp/tagger.ts` (LLM path) |
| Scale beyond local | `store/db.ts` (vec0 already; add sharding) |
| New ingestion | feed `observe()` more turns; add importers |

## Non-goals

POI/map-data acquisition, place search over the live world, hosted tier, and auth
are out of scope — openmap is the memory, fed by the agent's conversation.
