/** Prompt for pulling explicit place names out of free text. */
export function buildMentionsPrompt(text: string): string {
  return (
    "Extract names of physical places (restaurants, bars, shops, parks, venues) " +
    "explicitly mentioned. One name per line, no numbering. If none, return nothing.\n\nText:\n" +
    text
  );
}
