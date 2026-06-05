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
    '"constraints": {"openNow": boolean|null, "maxBudget": "low"|"mid"|"high"|null, "dietary": string[], "walkable": boolean|null}}\n' +
    `- goals: likely social purposes ONLY from: ${ALLOWED_GOALS.join(", ")} (a query is rarely literal: "romantic dinner" → ["date","romance"]).\n` +
    "- companions: alone|partner|kids|parents|family|friends|client, or null.\n" +
    "- concepts: lowercase single-word categories (coffee, ramen, bar, wine).\n" +
    "- vibe: lowercase adjectives for the feel (cozy, quiet, lively, romantic, outdoor, not-touristy, instagrammable).\n" +
    "- constraints: only what's implied; omit/none otherwise.\n\nQuery:\n" +
    text
  );
}
