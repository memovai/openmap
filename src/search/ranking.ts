import { type RememberedRow } from "../store/db.js";
import { type GeoPoint, type IntentFrame, type PersonaPrefs, type Place, type ScoredPlace, placeAttributeBlob, placeTextBlob } from "../core/types.js";
import { CONSTRAINT_MATCH_TERMS, RAW_NUMERIC_KEYS, constraintConceptTerms } from "../core/vocabulary.js";
import { geoAffinity } from "../core/geo.js";
import { dislikePenalty } from "../memory/persona.js";
import { derivePlaceVibe } from "../world/affordance.js";

// Ranking weights — explicit so scoring stays auditable, not magic.
export const W_REL = 0.6; // hybrid relevance (keyword+vector RRF)
export const W_AFFECT = 0.25; // how much the user loved the place
export const W_TASTE = 0.15; // alignment with the user's taste prior
export const W_GEO = 0.15; // proximity (scaled by the learned near-radius)
export const RRF_K = 60; // standard Reciprocal Rank Fusion constant

const round = (x: number, n = 3) => Number(x.toFixed(n));

/** Reciprocal Rank Fusion: merge ranked id-lists into one relevance score map.
 * Score = Σ 1/(k + rank). Avoids normalizing incompatible BM25 vs cosine scores. */
export function rrfMerge(lists: string[][], k = RRF_K): Map<string, number> {
  const m = new Map<string, number>();
  for (const list of lists) list.forEach((id, rank) => m.set(id, (m.get(id) ?? 0) + 1 / (k + rank + 1)));
  return m;
}

/**
 * Rank the user's remembered places. Relevance comes from **hybrid retrieval**
 * (keyword FTS/BM25 + vector cosine, fused by RRF) and is then blended with the
 * personal signals: felt affect, taste-prior alignment, matching affordances
 * (vibe), proximity (learned near-radius), and a dislike penalty.
 */
export function rankMemory(args: {
  items: RememberedRow[];
  relevance: Map<string, number>; // RRF scores by place id
  tasteSim: number[]; // taste-prior cosine per item (0 when unavailable)
  prefs: PersonaPrefs;
  beliefSignals?: RankingBeliefSignals;
  frame: IntentFrame;
  near: GeoPoint | null;
  nearRadiusKm: number;
  constraintLimits?: ConstraintLimits;
  limit: number;
}): ScoredPlace[] {
  const { items, relevance, tasteSim, prefs, beliefSignals, frame, near, nearRadiusKm, constraintLimits, limit } = args;
  if (items.length === 0) return [];
  const maxRel = Math.max(1e-9, ...[...relevance.values()]);
  const vibeSet = new Set(frame.vibe);

  const scored: ScoredPlace[] = items.map((it, i) => {
    const rel = (relevance.get(it.place.id) ?? 0) / maxRel; // normalized RRF [0,1]
    const [geo, dist] = geoAffinity(near, it.place, nearRadiusKm);
    const penalty = dislikePenalty(prefs, it.place);
    const placeVibe = derivePlaceVibe(it.place);
    const vibeHits = placeVibe.filter((v) => vibeSet.has(v)).length;
    const vibeBonus = 1 + Math.min(vibeHits, 3) * 0.15;
    const concepts = conceptAffinity(it.place, frame);
    const conceptBonus = 1 + Math.min(concepts.labels.length, 3) * 0.12;
    const goals = goalAffinity(it.place, frame);
    const goalBonus = 1 + goals.weight;
    const constraints = constraintAffinity(it.place, frame, dist, constraintLimits);
    const constraintBonus = 1 + constraints.hits * 0.18 - constraints.misses * 0.08;
    const symbolic = symbolicAffinity(it.place, frame, beliefSignals);
    const base = W_REL * rel + W_AFFECT * it.aggAffect + W_TASTE * (tasteSim[i] ?? 0) + W_GEO * geo;
    return {
      place: it.place,
      score: round(base * penalty * vibeBonus * conceptBonus * goalBonus * constraintBonus * symbolic.bonus, 4),
      distanceKm: dist == null ? null : round(dist),
      relationship: it.relationship,
      reasons: {
        relevance: round(rel), affect: round(it.aggAffect), tasteSim: round(tasteSim[i] ?? 0),
        geoAffinity: round(geo), vibeBonus: round(vibeBonus), conceptBonus: round(conceptBonus), goalBonus: round(goalBonus), constraintBonus: round(constraintBonus), symbolicBonus: round(symbolic.bonus), dislikePenalty: round(penalty),
        conceptHits: concepts.labels.join(",") || "none",
        goalHits: goals.labels.join(",") || "none",
        constraintHits: constraints.hitLabels.join(",") || "none", constraintMisses: constraints.missLabels.join(",") || "none",
        symbolicLikes: symbolic.likes.join(",") || "none",
        symbolicAvoids: symbolic.avoids.join(",") || "none",
        symbolicPursues: symbolic.pursues.join(",") || "none",
        placeVibe: placeVibe.join(",") || "none", goals: frame.goals.join(",") || "none",
      },
    };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

export interface ConstraintLimits {
  noiseMax?: number | null;
  crowdMax?: number | null;
  transitWalkMax?: number | null;
  walkTimeMax?: number | null;
}

export interface WeightedBeliefTerm {
  term: string;
  confidence: number;
}

export interface RankingBeliefSignals {
  likes: WeightedBeliefTerm[];
  avoids: WeightedBeliefTerm[];
  pursues: WeightedBeliefTerm[];
}

function conceptAffinity(place: Place, frame: IntentFrame): { labels: string[] } {
  const tokens = new Set(placeAttributeBlob(place).toLowerCase().match(/[a-z0-9_]+/g) ?? []);
  const labels: string[] = [];
  for (const concept of new Set(frame.concepts.map((c) => c.toLowerCase().trim()).filter(Boolean))) {
    if (tokens.has(concept)) labels.push(concept);
  }
  return { labels };
}

const GOAL_MATCH_WEIGHT = 0.12;

function goalAffinity(place: Place, frame: IntentFrame): { weight: number; labels: string[] } {
  const tags = new Set(place.tags.map((t) => t.toLowerCase().trim()).filter(Boolean));
  const activeGoals = supportedGoals(frame);
  const labels: string[] = [];
  let weight = 0;
  for (const goal of new Set(frame.goals.map((g) => g.toLowerCase().trim()).filter(Boolean))) {
    if (!activeGoals.has(goal)) continue;
    if (!tags.has(goal)) continue;
    labels.push(goal);
    weight += GOAL_MATCH_WEIGHT;
  }
  return { weight: Math.min(weight, 0.35), labels };
}

function symbolicAffinity(
  place: Place,
  frame: IntentFrame,
  signals: RankingBeliefSignals | undefined,
): { bonus: number; likes: string[]; avoids: string[]; pursues: string[] } {
  if (!signals) return { bonus: 1, likes: [], avoids: [], pursues: [] };
  const blob = placeAttributeBlob(place).toLowerCase();
  const tags = new Set(place.tags.map((t) => t.toLowerCase().trim()).filter(Boolean));
  const frameGoals = supportedGoals(frame);
  const relevantLikes = symbolicLikeTerms(frame);
  const placeMatches = (term: string) => {
    const t = term.toLowerCase().trim();
    return !!t && (tags.has(t) || blob.includes(t));
  };
  const likes = signals.likes.filter((s) => relevantLikes.has(s.term.toLowerCase().trim()) && placeMatches(s.term));
  const avoids = signals.avoids.filter((s) => placeMatches(s.term));
  const pursues = signals.pursues.filter((s) => {
    const term = s.term.toLowerCase().trim();
    return frameGoals.has(term) && placeMatches(term);
  });
  const plus = likes.reduce((n, s) => n + Math.min(1, s.confidence) * 0.12, 0) + pursues.reduce((n, s) => n + Math.min(1, s.confidence) * 0.08, 0);
  const minus = avoids.reduce((n, s) => n + Math.min(1, s.confidence) * 0.45, 0);
  return {
    bonus: Math.max(0.25, Math.min(1.4, 1 + plus - minus)),
    likes: likes.map((s) => s.term),
    avoids: avoids.map((s) => s.term),
    pursues: pursues.map((s) => s.term),
  };
}

function constraintAffinity(place: Place, frame: IntentFrame, distanceKm: number | null, limits: ConstraintLimits = {}): { hits: number; misses: number; hitLabels: string[]; missLabels: string[] } {
  const blob = placeAttributeBlob(place).toLowerCase();
  const has = (terms: string[]) => terms.some((t) => blob.includes(t));
  const rawNoise = rawNumber(place, RAW_NUMERIC_KEYS.noise);
  const rawNoiseDb = rawNumber(place, RAW_NUMERIC_KEYS.noiseDb);
  const noise = rawNoise ?? (rawNoiseDb == null ? null : Math.max(0, Math.min(1, (rawNoiseDb - 35) / 50)));
  const crowd = rawNumber(place, RAW_NUMERIC_KEYS.crowd);
  const transitWalk = rawNumber(place, RAW_NUMERIC_KEYS.transitWalk);
  const walkTime = rawNumber(place, RAW_NUMERIC_KEYS.walkTime);
  const quiet = noise != null ? noise <= (limits.noiseMax ?? 0.35) : has(CONSTRAINT_MATCH_TERMS.noise.quiet);
  const loud = noise != null ? noise > (limits.noiseMax ?? 0.35) : has(CONSTRAINT_MATCH_TERMS.noise.loud);
  const lowCrowd = crowd != null ? crowd <= (limits.crowdMax ?? 0.35) : has(CONSTRAINT_MATCH_TERMS.crowd.low);
  const highCrowd = crowd != null ? crowd > (limits.crowdMax ?? 0.35) : has(CONSTRAINT_MATCH_TERMS.crowd.high);
  const transitOk = transitWalk != null ? transitWalk <= (limits.transitWalkMax ?? 8) : has(CONSTRAINT_MATCH_TERMS.travelMode.transit);
  const walkableOk = walkTime != null ? walkTime <= (limits.walkTimeMax ?? 10) : has(CONSTRAINT_MATCH_TERMS.walkable) || (distanceKm != null && distanceKm <= 1.5);
  const hitLabels: string[] = [];
  const missLabels: string[] = [];
  const check = (active: boolean, label: string, ok: boolean) => {
    if (!active) return;
    (ok ? hitLabels : missLabels).push(label);
  };

  check(frame.constraints.openNow === true, "open", has(CONSTRAINT_MATCH_TERMS.openNow));
  check(frame.constraints.walkable === true, "walkable", walkableOk);
  check(frame.constraints.noise === "quiet", "noise:quiet", quiet && !loud);
  check(frame.constraints.noise === "loud", "noise:loud", loud);
  check(frame.constraints.crowd === "low", "crowd:low", lowCrowd || (quiet && !highCrowd));
  check(frame.constraints.crowd === "high", "crowd:high", highCrowd);
  check(frame.constraints.travelMode === "transit", "transit", transitOk);
  check(frame.constraints.travelMode === "drive", "drive", has(CONSTRAINT_MATCH_TERMS.travelMode.drive));
  for (const d of frame.constraints.dietary ?? []) {
    const term = d.toLowerCase();
    check(Boolean(term), term, has([term, ...CONSTRAINT_MATCH_TERMS.dietary]));
  }
  check(frame.constraints.maxBudget === "low", "cheap", has(CONSTRAINT_MATCH_TERMS.budget.low));
  check(frame.constraints.maxBudget === "high", "fancy", has(CONSTRAINT_MATCH_TERMS.budget.high));
  return { hits: hitLabels.length, misses: missLabels.length, hitLabels, missLabels };
}

function supportedGoals(frame: IntentFrame): Set<string> {
  const rawTokens = tokens(frame.rawQuery);
  const frameTags = new Set([...frame.concepts, ...frame.vibe].map((t) => t.toLowerCase().trim()).filter(Boolean));
  const supported = new Set<string>();
  for (const goal of frame.goals.map((g) => g.toLowerCase().trim()).filter(Boolean)) {
    if (frameTags.has(goal) || rawTokens.some((t) => tokenSimilar(t, goal))) supported.add(goal);
  }
  return supported;
}

function tokens(text: string): string[] {
  return [...new Set(text.toLowerCase().match(/[a-z0-9]+/g) ?? [])].filter((t) => t.length > 1);
}

function tokenSimilar(a: string, b: string): boolean {
  if (a === b) return true;
  const n = Math.min(a.length, b.length);
  return n >= 5 && a.slice(0, 5) === b.slice(0, 5);
}

function symbolicLikeTerms(frame: IntentFrame): Set<string> {
  const terms = new Set([...frame.concepts, ...frame.vibe, ...constraintConceptTerms(frame.constraints)].map((t) => t.toLowerCase().trim()).filter(Boolean));
  for (const goal of supportedGoals(frame)) terms.add(goal);
  return terms;
}

function rawNumber(place: Place, keys: string[]): number | null {
  for (const key of keys) {
    const v = place.raw[key];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() && Number.isFinite(Number(v))) return Number(v);
  }
  return null;
}
