import { type Place, type PersonaPrefs, placeTextBlob } from "../core/types.js";

/** Positive free text describing what a user likes — embedded and blended into
 * the taste vector so an explicit persona steers discovery even cold. */
export function personaPositiveText(prefs: PersonaPrefs): string {
  return [
    ...prefs.likes,
    ...prefs.vibes,
    ...prefs.dietary,
    prefs.budget ? `${prefs.budget} budget` : "",
    prefs.notes ?? "",
  ]
    .filter((s) => s && s.trim())
    .join(", ");
}

/** Multiplicative penalty in (0,1] for a candidate matching a disliked term. */
export function dislikePenalty(prefs: PersonaPrefs, place: Place, factor = 0.4): number {
  if (prefs.dislikes.length === 0) return 1;
  const blob = placeTextBlob(place).toLowerCase();
  const hit = prefs.dislikes.some((d) => {
    const t = d.trim().toLowerCase();
    return t.length > 0 && blob.includes(t);
  });
  return hit ? factor : 1;
}
