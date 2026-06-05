import { type RememberedRow } from "../store/db.js";
import { type GeoPoint, type IntentFrame, type PersonaPrefs, type ScoredPlace } from "../core/types.js";
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
  frame: IntentFrame;
  near: GeoPoint | null;
  nearRadiusKm: number;
  limit: number;
}): ScoredPlace[] {
  const { items, relevance, tasteSim, prefs, frame, near, nearRadiusKm, limit } = args;
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
    const base = W_REL * rel + W_AFFECT * it.aggAffect + W_TASTE * (tasteSim[i] ?? 0) + W_GEO * geo;
    return {
      place: it.place,
      score: round(base * penalty * vibeBonus, 4),
      distanceKm: dist == null ? null : round(dist),
      relationship: it.relationship,
      reasons: {
        relevance: round(rel), affect: round(it.aggAffect), tasteSim: round(tasteSim[i] ?? 0),
        geoAffinity: round(geo), vibeBonus: round(vibeBonus), dislikePenalty: round(penalty),
        placeVibe: placeVibe.join(",") || "none", goals: frame.goals.join(",") || "none",
      },
    };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}
