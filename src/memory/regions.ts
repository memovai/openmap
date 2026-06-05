import { type DB } from "../store/db.js";

const round = (x: number, n = 5) => Number(x.toFixed(n));

/** An area the user is active in — the user↔area relationship, derived as an
 * aggregate over their place activity (distinct from individual place memories
 * and from objective geo facts; linked to both via coords + placeIds). */
export interface Area {
  lat: number;
  lng: number;
  radiusKm: number;
  count: number; // distinct remembered places in this area = activity weight
  placeIds: string[];
}

/**
 * Cluster the user's remembered places (those with coords) into frequented
 * areas via a simple geographic grid. Captures "where this user spends time" —
 * so picking a place near X strengthens X as one of the user's areas. Empty
 * until the host supplies coords (conversation-only: no external geocoding).
 */
export function frequentedAreas(db: DB, userId: string, opts: { cellKm?: number; limit?: number } = {}): Area[] {
  const cellKm = opts.cellKm ?? 2;
  const cellDeg = cellKm / 111; // ~111 km per degree (good enough for clustering)
  const cells = new Map<string, { lat: number; lng: number; ids: string[] }>();
  for (const { place } of db.iterRemembered(userId)) {
    if (place.lat == null || place.lng == null) continue;
    const key = `${Math.round(place.lat / cellDeg)}:${Math.round(place.lng / cellDeg)}`;
    const c = cells.get(key) ?? { lat: 0, lng: 0, ids: [] };
    c.lat += place.lat;
    c.lng += place.lng;
    c.ids.push(place.id);
    cells.set(key, c);
  }
  const areas: Area[] = [...cells.values()].map((c) => ({
    lat: round(c.lat / c.ids.length),
    lng: round(c.lng / c.ids.length),
    radiusKm: cellKm,
    count: c.ids.length,
    placeIds: c.ids,
  }));
  areas.sort((a, b) => b.count - a.count);
  return areas.slice(0, opts.limit ?? 5);
}

/** The user's most-active area centroid (best default "near me" when no home). */
export function primaryArea(db: DB, userId: string): { lat: number; lng: number } | null {
  const top = frequentedAreas(db, userId, { limit: 1 })[0];
  return top ? { lat: top.lat, lng: top.lng } : null;
}
