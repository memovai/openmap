# openmap — Memory Model

How openmap models map-aware memory. Its only input is the agent's conversation;
places and attributes are extracted from what's said (no external POI lookup).
(Prior art that informed the design is acknowledged in the README references.)

## Principle

Raw activity rises into distilled belief, and vague words get a personal meaning.
"I searched for coffee" (episodic) becomes "I probably like coffee" (semantic);
"near" becomes "≈3 km for this user" (calibration). Memory is per-user; the only
shared thing is the set of canonical place nodes referenced by id.

## Layers

### L0 — Events (episodic, per user)
Every `remember`, `observe`, and `recall` logs an event: `kind`, text, optional
place, extracted `concepts` + `intents`. The behavioral substrate. Searching is
conversation too, so queries feed inference.

### L1/L2 — Beliefs (semantic knowledge graph, per user)
Triples `(subject, predicate, object)` with **confidence + provenance + source**:
`user —likes→ coffee` (0.7, from N events), `user —lives_near→ place`,
`user —avoids→ loud`, `user —pursues→ date`. Promoted from events by
`consolidate()`; reconciled (ADD / UPDATE / NOOP); inferred beliefs **decay** with
recency, stated ones don't (stated > inferred).

### L3 — Persona (per user)
The distilled profile = stated preferences merged with derived ones (top beliefs).

### Spatial self-model (per user) — the layer that makes it *map-aware*
Kept distinct from objective geo facts, linked at query time:
- **Calibrations** — the personal meaning of fuzzy terms, learned from accepted
  options: `near` (km, tolerance), `walk_time` (min), `budget` (typical spend),
  `noise` (level). Adding a term is one registry entry — no hardcoded logic.
- **Anchors** — home/work (stated), usual area.
- **Frequented areas** — clusters of place activity = the *user↔area* relationship;
  default "near me" resolves to the most-active area.

## The flagship loops

**"Do I like coffee?"** — check beliefs; if absent, drill down to events + loved
places, compute a saturating confidence with provenance, and consolidate up.

**"A cozy date spot"** — resolve the query into an intent frame
{goals, companions, occasion, vibe, constraints}; search the *resolved frame* (not
the literal words) over remembered places, ranked by taste prior + matching
affordances + proximity (using the learned near-radius).

**Auto-calibration (revealed preference)** — when `observe` sees the agent offer
options with distances/prices and the user accept one, it learns the accepted
measure: picked a place 3 km away → `near ≥ 3 km`; rejections ("too far") are
skipped.

## Reconcile (ADD / UPDATE / NOOP)

A new (place, relationship) observation is reconciled against existing memory:
new place → ADD; "want_to_go" then actually visited → UPDATE (not a duplicate);
sentiment flip (loved → disliked) → UPDATE (latest wins); same → NOOP.

## Why a graph, not just vectors

Vectors answer "what's similar"; they can't answer "where's home", "who did I go
with", or "what does *near* mean to me". Those are relations and learned scalars —
first-class in the belief graph and the calibration layer. Vectors are used for
*retrieval* of nodes, not as the model itself.

## Repository layout

```
core/   types · geo · config
store/  db.ts (SQLite + sqlite-vec)
nlp/    embedding · extract · tagger
memory/ inference · taste · persona · anchors · regions · calibration · graph
world/  affordance · relations
search/ ranking
openmap.ts (facade) · cli.ts · mcp.ts · index.ts
```
