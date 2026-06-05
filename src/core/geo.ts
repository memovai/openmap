import { type GeoPoint, type Place } from "./types.js";

export function haversineKm(a: GeoPoint, lat: number, lng: number): number {
  const r = 6371.0;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dphi = toRad(lat - a.lat);
  const dlmb = toRad(lng - a.lng);
  const h =
    Math.sin(dphi / 2) ** 2 +
    Math.cos(toRad(a.lat)) * Math.cos(toRad(lat)) * Math.sin(dlmb / 2) ** 2;
  return 2 * r * Math.asin(Math.sqrt(h));
}

/** Additive proximity bonus — recall must not *exclude* far places. */
export function geoBonus(near: GeoPoint | null, p: Place, weight: number): [number, number | null] {
  if (!near || p.lat == null || p.lng == null) return [0, null];
  const dist = haversineKm(near, p.lat, p.lng);
  return [weight / (1 + dist / 2), dist];
}

/** Multiplicative distance gate — treats 'near me' as a hard intent.
 * ~1 nearby, decaying to ~0 far away. Unknown coords → neutral 0.5. */
export function geoGate(near: GeoPoint | null, p: Place, decayKm: number): [number, number | null] {
  if (!near) return [1, null];
  if (p.lat == null || p.lng == null) return [0.5, null];
  const dist = haversineKm(near, p.lat, p.lng);
  return [Math.exp(-dist / decayKm), dist];
}

/** Proximity affinity scaled by the user's *learned* near-radius (not a hardcoded
 * constant): ~1 well inside the radius, ~0.37 at the radius, decaying beyond. */
export function geoAffinity(near: GeoPoint | null, p: Place, radiusKm: number): [number, number | null] {
  if (!near || p.lat == null || p.lng == null) return [0, null];
  const dist = haversineKm(near, p.lat, p.lng);
  return [Math.exp(-dist / Math.max(radiusKm, 0.1)), dist];
}
