import { type DB } from "../store/db.js";
import { type GeoPoint, type Place, type Predicate } from "../core/types.js";
import { haversineKm } from "../core/geo.js";
import { nearRadiusKm } from "./calibration.js";
import { primaryArea } from "./regions.js";

const round = (x: number, n = 3) => Number(x.toFixed(n));

export interface Anchors {
  home: Place | null;
  work: Place | null;
  usualArea: { lat: number; lng: number; radiusKm: number } | null;
  /** The user's learned meaning of "near", in km (subjective spatial semantics). */
  nearRadiusKm: number;
}

/** A user's geographic anchors, learned from memory: home/work (stated place
 * roles) + a "usual area" (centroid + radius of remembered places). Lets
 * unanchored queries ("somewhere for dinner") default to where the user is. */
export function computeAnchors(db: DB, userId: string): Anchors {
  const placeOf = (pred: Predicate) => {
    const b = db.listBeliefs(userId, { predicate: pred })[0];
    return b ? db.getPlace(b.object) : null;
  };
  const pts = db.iterRemembered(userId).map((r) => r.place).filter((p) => p.lat != null && p.lng != null);
  let usualArea: Anchors["usualArea"] = null;
  if (pts.length) {
    const lat = pts.reduce((s, p) => s + p.lat!, 0) / pts.length;
    const lng = pts.reduce((s, p) => s + p.lng!, 0) / pts.length;
    const radiusKm = round(Math.max(...pts.map((p) => haversineKm({ lat, lng }, p.lat!, p.lng!))));
    usualArea = { lat: round(lat, 5), lng: round(lng, 5), radiusKm };
  }
  return { home: placeOf("lives_near"), work: placeOf("works_near"), usualArea, nearRadiusKm: nearRadiusKm(db, userId) };
}

/** Best default location when a query gives none:
 * home → most-active area → global usual-area centroid → null. */
export function defaultAnchor(db: DB, userId: string): GeoPoint | null {
  const a = computeAnchors(db, userId);
  if (a.home?.lat != null && a.home.lng != null) return { lat: a.home.lat, lng: a.home.lng };
  const top = primaryArea(db, userId);
  if (top) return top;
  if (a.usualArea) return { lat: a.usualArea.lat, lng: a.usualArea.lng };
  return null;
}
