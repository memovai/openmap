/** Prompt for resolving a maps query into its latent IntentFrame. */
export const ALLOWED_GOALS = [
  "date",
  "romance",
  "work",
  "study",
  "family",
  "celebration",
  "business",
  "hangout",
  "solo",
  "explore",
];

export function buildIntentPrompt(text: string): string {
  return (
    "Resolve the latent intent behind a maps/places query. Return strict JSON:\n" +
    '{"goals": string[], "companions": string|null, "occasion": string|null, ' +
    '"concepts": string[], "vibe": string[], ' +
    '"constraints": {"openNow": boolean|null, "maxBudget": "low"|"mid"|"high"|null, "dietary": string[], "walkable": boolean|null, "noise": "quiet"|"moderate"|"loud"|null, "crowd": "low"|"moderate"|"high"|null, "travelMode": "walk"|"transit"|"drive"|null}}\n' +
    `- goals: likely social purposes ONLY from: ${ALLOWED_GOALS.join(", ")} (a query is rarely literal: "romantic dinner" → ["date","romance"]).\n` +
    "- companions: alone|partner|kids|parents|family|friends|client, or null.\n" +
    "- concepts: lowercase single-word categories or constraints (coffee, ramen, bar, wine, vegetarian, open_late, walkable, low_crowd, crowded, transit, parking).\n" +
    "- vibe: lowercase adjectives for the feel (cozy, quiet, loud, lively, romantic, outdoor, not-touristy, instagrammable).\n" +
    "- constraints: set openNow=true for open now/open late/late-night; walkable=true and travelMode=walk for walkable/on foot; travelMode=transit for near station/subway/train; travelMode=drive for parking/driving. Set noise=quiet for quiet/not noisy, crowd=low for not crowded/uncrowded, dietary for vegetarian/vegan. Omit/none otherwise.\n\nQuery:\n" +
    text
  );
}
