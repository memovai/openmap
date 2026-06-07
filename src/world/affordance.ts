import { type Place, placeAttributeBlob } from "../core/types.js";
import { VIBE_CONCEPTS, extractConcepts } from "../nlp/extract.js";

/** Affordance/vibe tags for a place, derived from attribute text (category/tags/raw).
 * Sparse for raw OSM; richer when the provider supplies review/description text.
 * This is the place-side affordance layer that indirect intent matches against. */
export function derivePlaceVibe(place: Place): string[] {
  return extractConcepts(placeAttributeBlob(place)).filter((c) => VIBE_CONCEPTS.has(c));
}
