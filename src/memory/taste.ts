import { type DB } from "../store/db.js";
import { type Embedder, blendVectors } from "../nlp/embedding.js";
import { type PersonaPrefs } from "../core/types.js";
import { personaPositiveText } from "./persona.js";

/** Affect-weighted centroid of a user's loved/visited place embeddings = the
 * derived taste prior used to disambiguate indirect queries. */
export function tasteVector(db: DB, userId: string): Float32Array | null {
  const rows = db.iterRemembered(userId).filter((r) => r.aggAffect > 0 && r.embedding);
  if (rows.length === 0) return null; // no embedder / no embedded loved places
  const dim = rows[0]!.embedding!.length;
  const centroid = new Float32Array(dim);
  let wsum = 0;
  for (const { embedding, aggAffect } of rows) {
    if (!embedding || embedding.length !== dim) continue;
    for (let i = 0; i < dim; i++) centroid[i]! += embedding[i]! * aggAffect;
    wsum += aggAffect;
  }
  if (wsum === 0) return null;
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    centroid[i]! /= wsum;
    norm += centroid[i]! * centroid[i]!;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) centroid[i]! /= norm;
  return centroid;
}

/** Derived taste blended with the explicit persona's positive preferences, so a
 * brand-new user with only a stated persona still gets taste-aware results. */
export async function effectiveTaste(
  db: DB,
  embedder: Embedder | null,
  userId: string,
  prefs: PersonaPrefs,
): Promise<Float32Array | null> {
  const derived = tasteVector(db, userId);
  const posText = personaPositiveText(prefs);
  if (!posText || !embedder) return derived; // no embedder → derived-only (or null)
  return blendVectors(derived, await embedder.embedOne(posText));
}
