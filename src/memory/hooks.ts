/**
 * Agent-integration hooks — the layer a host agent (OpenClaw, Claude Code, …)
 * actually wires into its turn loop. Mirrors the two hooks a memory layer needs:
 *
 *  - auto-recall  → before the agent answers, build an injectable context block
 *                   (stable persona for the system prompt + relevant places to
 *                   prepend to the user message).
 *  - auto-capture → after a turn, persist the raw exchange (L0, for grounding)
 *                   and distil it into structured memory.
 *
 * These only format/orchestrate; all extraction + retrieval lives in OpenMap.
 */
import { type Persona, type ScoredPlace } from "../core/types.js";
import { type Anchors } from "./anchors.js";
import { type Calibration } from "./calibration.js";

export interface RecallBlockSource {
  turnId: number;
  role: string;
  at: string | null;
  snippet: string;
}
export type RecallBlockSources = Record<string, RecallBlockSource[]>;

/** Persona block — stable across a session, so a host caches it on the system
 * prompt. Empty string when nothing is known yet (don't inject noise). */
export function formatPersonaContext(persona: Persona, anchors: Anchors, calibrations: Calibration[]): string {
  const lines: string[] = [];
  const likes = uniq([...persona.stated.likes, ...persona.derived.likes]);
  const avoids = uniq([...persona.stated.dislikes, ...persona.derived.avoids]);
  const pursues = uniq(persona.derived.pursues);
  if (likes.length) lines.push(`- Likes: ${likes.join(", ")}`);
  if (avoids.length) lines.push(`- Avoids: ${avoids.join(", ")}`);
  if (pursues.length) lines.push(`- Wants to try: ${pursues.join(", ")}`);
  if (persona.stated.vibes.length) lines.push(`- Preferred vibe: ${persona.stated.vibes.join(", ")}`);
  if (persona.stated.dietary.length) lines.push(`- Dietary: ${persona.stated.dietary.join(", ")}`);
  if (persona.stated.budget) lines.push(`- Budget: ${persona.stated.budget}`);
  if (persona.stated.notes) lines.push(`- Note: ${persona.stated.notes}`);

  // Geography only when there's a real anchor — a default "near" radius alone is
  // generic, not something we actually learned about this user.
  const geo: string[] = [];
  if (anchors.home) geo.push(`home ${anchors.home.name}`);
  if (anchors.work) geo.push(`work ${anchors.work.name}`);
  if (anchors.usualArea) geo.push(`usually around (${anchors.usualArea.lat.toFixed(3)}, ${anchors.usualArea.lng.toFixed(3)})`);
  if (geo.length) lines.push(`- Location: ${geo.join("; ")}; "near" ≈ ${anchors.nearRadiusKm}km`);

  // only terms we actually learned from this user (priors carry samples=0)
  const cal = calibrations.filter((c) => c.value != null && c.samples > 0 && c.term !== "near");
  if (cal.length) lines.push(`- Learned terms: ${cal.map((c) => `${c.term} ≈ ${c.value}${c.unit}`).join(", ")}`);

  if (!lines.length) return "";
  return `<user-place-profile>\nWhat I know about this user's taste & geography:\n${lines.join("\n")}\n</user-place-profile>`;
}

/** Relevant-places block — dynamic per turn, prepended to the user message. */
export function formatRecallBlock(places: ScoredPlace[], sources: RecallBlockSources = {}): string {
  if (!places.length) return "";
  const rows = places.map((p) => {
    const bits: string[] = [];
    if (p.relationship && p.relationship !== "mentioned") bits.push(p.relationship);
    if (p.place.tags?.length) bits.push(p.place.tags.slice(0, 4).join("/"));
    if (p.distanceKm != null) bits.push(`${p.distanceKm}km away`);
    const src = sources[p.place.id]?.[0];
    if (src) bits.push(`source turn#${src.turnId}`);
    const meta = bits.length ? ` (${bits.join("; ")})` : "";
    return `- ${p.place.name}${meta}`;
  });
  return (
    "<recalled-places>\n" +
    "Places from this user's memory that may be relevant — prefer these when they fit; verify before stating as fact:\n" +
    rows.join("\n") +
    "\n</recalled-places>"
  );
}

function uniq(xs: string[]): string[] {
  return [...new Set(xs.map((x) => x.trim()).filter(Boolean))];
}
