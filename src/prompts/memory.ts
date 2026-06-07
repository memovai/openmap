/** Prompt + constants for structured per-place memory extraction from a turn. */
import { RELATIONSHIPS } from "../core/types.js";
import { CONCEPT_LEXICON, MEASURE_TERMS, VIBE_CONCEPTS } from "../core/vocabulary.js";

export const MEMORY_RELATIONSHIPS = [...RELATIONSHIPS];

const conceptExamples = Object.keys(CONCEPT_LEXICON).filter((c) => !VIBE_CONCEPTS.has(c)).slice(0, 10).join(", ");
const vibeExamples = [...VIBE_CONCEPTS].join(", ");

export function buildMemoryPrompt(text: string, context: string): string {
  return (
    "Extract the physical places the USER refers to in their message. Use the assistant's prior message only as context for distances/prices. Return strict JSON:\n" +
    `{"places":[{"name":string,"relationship":"${MEMORY_RELATIONSHIPS.join("|")}","companions":string[],"region":string|null,"measures":[{"term":"${MEASURE_TERMS.join("|")}","value":number,"evidence":string}],"goal":string|null,"concepts":[{"tag":string,"evidence":string}]}]}\n` +
    "- Per-place sentiment (the user may love one and dislike another in the same message).\n" +
    `- concepts: lowercase category/vibe/constraint words describing THIS place. Use the user's own evidence phrase for each tag; prefer canonical tags like ${conceptExamples}, ${vibeExamples} when grounded, and allow new grounded tags when needed (e.g. 'shoulder-to-shoulder' -> {"tag":"crowded","evidence":"shoulder-to-shoulder"}).\n` +
    "- Only places the USER narrates; skip the assistant's suggestions they didn't accept.\n" +
    "- region: a named area the place is in (e.g. Shibuya), else null. measures: attach distance(km)/walk(min)/price/noise/crowd/transit only when THIS place's evidence supports it; include the exact evidence phrase. Do not invent default 0.5 values. noise and crowd are 0..1 where 0 is quiet/empty and 1 is loud/packed; transit_walk is minutes from station/subway/metro/train/bus. goal: the social purpose if any (date, work, family…), else null.\n\n" +
    `Assistant (context): ${context}\nUser: ${text}`
  );
}
