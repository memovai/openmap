/** Prompt for resolving a maps query into its latent IntentFrame. */
import { ALLOWED_GOALS, CONCEPT_LEXICON, VIBE_CONCEPTS } from "../core/vocabulary.js";

export { ALLOWED_GOALS } from "../core/vocabulary.js";

const conceptExamples = Object.keys(CONCEPT_LEXICON).filter((c) => !VIBE_CONCEPTS.has(c)).slice(0, 12).join(", ");
const vibeExamples = [...VIBE_CONCEPTS].join(", ");

export function buildIntentPrompt(text: string): string {
  return (
    "Resolve the latent intent behind a maps/places query. Return strict JSON:\n" +
    '{"goals": string[], "companions": string|null, "occasion": string|null, ' +
    '"concepts": string[], "vibe": string[], ' +
    '"constraints": {"openNow": boolean|null, "maxBudget": "low"|"mid"|"high"|null, "dietary": string[], "walkable": boolean|null, "noise": "quiet"|"moderate"|"loud"|null, "crowd": "low"|"moderate"|"high"|null, "travelMode": "walk"|"transit"|"drive"|null}}\n' +
    `- goals: likely social purposes ONLY from: ${ALLOWED_GOALS.join(", ")} (a query is rarely literal: "romantic dinner" → ["date","romance"]).\n` +
    "- companions: alone|partner|kids|parents|family|friends|client, or null.\n" +
    `- concepts: lowercase category/constraint tags, such as ${conceptExamples}; new grounded tags are allowed.\n` +
    `- vibe: lowercase adjectives for the feel, such as ${vibeExamples}; new grounded vibe tags are allowed.\n` +
    "- constraints: infer practical requirements semantically from the user's wording, not by exact keyword matching. Use openNow, walkable, travelMode, dietary, budget, ambient noise, and crowding only when the query meaning clearly asks for them; omit/none otherwise.\n\nQuery:\n" +
    text
  );
}
