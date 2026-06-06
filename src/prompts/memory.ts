/** Prompt + constants for structured per-place memory extraction from a turn. */
export const MEMORY_RELATIONSHIPS = ["loved", "liked", "visited", "want_to_go", "disliked", "mentioned"];

export function buildMemoryPrompt(text: string, context: string): string {
  return (
    "Extract the physical places the USER refers to in their message. Use the assistant's prior message only as context for distances/prices. Return strict JSON:\n" +
    '{"places":[{"name":string,"relationship":"loved|liked|visited|want_to_go|disliked|mentioned","companions":string[],"region":string|null,"measures":[{"term":"near|walk_time|budget|noise|crowd|transit_walk","value":number}],"goal":string|null,"concepts":string[]}]}\n' +
    "- Per-place sentiment (the user may love one and dislike another in the same message).\n" +
    "- concepts: lowercase category/vibe/constraint words describing THIS place; prefer canonical words like coffee, ramen, bar, vegetarian, open_late, walkable, low_crowd, crowded, transit, parking, quiet, loud, cozy, romantic, outdoor, cheap, fancy (e.g. a 'quiet coffee spot' → [\"coffee\",\"quiet\"]).\n" +
    "- Only places the USER narrates; skip the assistant's suggestions they didn't accept.\n" +
    "- region: a named area the place is in (e.g. Shibuya), else null. measures: attach distance(km)/walk(min)/price; noise and crowd are 0..1 where 0 is quiet/empty and 1 is loud/packed; transit_walk is minutes from station/subway/metro/train/bus. goal: the social purpose if any (date, work, family…), else null.\n\n" +
    `Assistant (context): ${context}\nUser: ${text}`
  );
}
