import { type DB } from "../store/db.js";
import { type Place } from "../core/types.js";
import { haversineKm } from "../core/geo.js";

const round = (x: number, n = 3) => Number(x.toFixed(n));

export interface RelatedPlace {
  place: Place;
  relations: string[];
  distanceKm: number | null;
  similarity: number | null;
}

/** Place↔place relations from the shared world model: `similar` via vector KNN
 * over stored embeddings, `near` via geo distance. */
export function relatedPlaces(
  db: DB,
  placeId: string,
  opts: { limit?: number; radiusKm?: number } = {},
): RelatedPlace[] {
  const target = db.getPlace(placeId);
  if (!target) return [];
  const limit = opts.limit ?? 5;
  const radius = opts.radiusKm ?? 2;
  const map = new Map<string, RelatedPlace>();

  const tEmb = db.embeddingFor(placeId);
  if (tEmb) {
    for (const { placeId: pid, score } of db.searchPlaceVectors(tEmb, limit + 6)) {
      if (pid === placeId || score < 0.6) continue;
      const p = db.getPlace(pid);
      if (p) map.set(pid, { place: p, relations: ["similar"], distanceKm: null, similarity: round(score) });
    }
  }
  if (target.lat != null && target.lng != null) {
    for (const { place } of db.allPlacesWithEmbeddings()) {
      if (place.id === placeId || place.lat == null || place.lng == null) continue;
      const d = round(haversineKm({ lat: target.lat, lng: target.lng }, place.lat, place.lng));
      if (d > radius) continue;
      const ex = map.get(place.id);
      if (ex) {
        ex.relations.push("near");
        ex.distanceKm = d;
      } else map.set(place.id, { place, relations: ["near"], distanceKm: d, similarity: null });
    }
  }
  return [...map.values()].sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0)).slice(0, limit);
}
