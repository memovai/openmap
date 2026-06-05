import { type Place, placeTextBlob } from "../core/types.js";
import { VIBE_CONCEPTS, extractConcepts } from "../nlp/extract.js";

/** Affordance/vibe tags for a place, derived from its text (name/category/tags).
 * Sparse for raw OSM; richer when the provider supplies review/description text.
 * This is the place-side affordance layer that indirect intent matches against. */
export function derivePlaceVibe(place: Place): string[] {
  return extractConcepts(placeTextBlob(place)).filter((c) => VIBE_CONCEPTS.has(c));
}
