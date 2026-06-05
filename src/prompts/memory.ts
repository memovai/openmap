/** Prompt + constants for structured per-place memory extraction from a turn. */
export const MEMORY_RELATIONSHIPS = ["loved", "liked", "visited", "want_to_go", "disliked", "mentioned"];

export function buildMemoryPrompt(text: string, context: string): string {
  return (
    "Extract the physical places the USER refers to in their message. Use the assistant's prior message only as context for distances/prices. Return strict JSON:\n" +
    '{"places":[{"name":string,"relationship":"loved|liked|visited|want_to_go|disliked|mentioned","companions":string[],"region":string|null,"measures":[{"term":"near|walk_time|budget","value":number}],"goal":string|null,"concepts":string[]}]}\n' +
    "- Per-place sentiment (the user may love one and dislike another in the same message).\n" +
    "- concepts: lowercase category/vibe words describing THIS place (e.g. a 'quiet coffee spot' → [\"coffee\",\"quiet\"]).\n" +
    "- Only places the USER narrates; skip the assistant's suggestions they didn't accept.\n" +
    "- region: a named area the place is in (e.g. Shibuya), else null. measures: attach the distance(km)/walk(min)/price the chosen place was described with. goal: the social purpose if any (date, work, family…), else null.\n\n" +
    `Assistant (context): ${context}\nUser: ${text}`
  );
}
