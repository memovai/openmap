import { type Config } from "../core/config.js";
import { type LLMRunner } from "./llm.js";
import { buildMentionsPrompt } from "../prompts/mentions.js";
import { CONCEPT_LEXICON, CONSTRAINT_MATCH_TERMS, INTENT_LEXICON, TAG_TO_CONCEPT, VIBE_CONCEPTS } from "../core/vocabulary.js";

export { VIBE_CONCEPTS } from "../core/vocabulary.js";

// ===========================================================================
// 1. Mention extraction — free text → candidate place names (for resolution)
// ===========================================================================
export interface Extractor {
  extract(text: string): Promise<string[]>;
}

// Double-style quotes only — a straight apostrophe here would false-match
// contractions (e.g. "let's do X" → "s do X").
const QUOTED = /["“”]([^"“”]{2,60})["“”]/g;
const PROPER = /\b(\p{Lu}[\p{L}\p{N}&'.-]+(?:\s+(?:\p{Lu}[\p{L}\p{N}&'.-]+|(?:de|la|le|du|of|the|and)\b|&)){0,4})/gu;
const STOP_LEADING = new Set([
  "I", "We", "They", "My", "The", "A", "An", "Last", "This", "That", "For", "After", "Before",
  "Date", "Dinner", "Lunch", "Brunch", "Breakfast",
]);
const COMMAND_LEADING = new Set([
  "Pick",
  "Choose",
  "Try",
  "Tried",
  "Visit",
  "Visited",
  "Book",
  "Reserve",
  "Skip",
  "Love",
  "Loved",
  "Like",
  "Liked",
  "Dislike",
  "Disliked",
]);

function cleanProperName(span: string): string | null {
  const words = span.trim().split(/\s+/).filter(Boolean);
  while (words.length > 0 && COMMAND_LEADING.has(words[0]!.replace(/[^\p{L}]/gu, ""))) words.shift();
  if (words.length === 0) return null;
  if (words.length === 1 && STOP_LEADING.has(words[0]!)) return null;
  return words.join(" ");
}

function dedupe(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const it of items) {
    const k = it.toLowerCase();
    if (!seen.has(k)) {
      seen.add(k);
      out.push(it);
    }
  }
  return out;
}

const NULL_MENTION_RE = /^(?:none|null|nil|n\/a|nothing(?:\s+to\s+extract)?|no\s+(?:explicit\s+)?(?:physical\s+)?places?(?:\s+(?:mentioned|found|to\s+extract))?\.?)$/i;

function stripFence(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  return (fenced ? fenced[1]! : text).trim();
}

function cleanLLMMentionName(raw: unknown): string | null {
  let name = String(raw ?? "").trim();
  name = name.replace(/^[-•*\s\t]+/, "").replace(/^\d+[\s.)-]+/, "").trim();
  name = name.replace(/^(?:name|place|places?)\s*:\s*/i, "").trim();
  name = name.replace(/^["'“”`]+|["'“”`.,;:]+$/g, "").trim();
  if (!name || NULL_MENTION_RE.test(name)) return null;
  if (/^no\b.*\bplaces?\b.*\b(?:mentioned|found|extract)\b/i.test(name)) return null;
  if (/^[\[{].*[\]}]$/.test(name)) return null;
  if (name.length > 80) return null;
  return name;
}

function namesFromJson(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((item) =>
      typeof item === "object" && item !== null && "name" in item
        ? [String((item as { name?: unknown }).name ?? "")]
        : [String(item ?? "")],
    );
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    for (const key of ["places", "names", "place_names"]) {
      if (Array.isArray(obj[key])) return namesFromJson(obj[key]);
    }
  }
  return [];
}

function parseLLMMentionResponse(content: string): { names: string[]; explicitNone: boolean } {
  const body = stripFence(content);
  if (!body) return { names: [], explicitNone: false };
  if (NULL_MENTION_RE.test(body) || /^no\b.*\bplaces?\b.*\b(?:mentioned|found|extract)\b/i.test(body))
    return { names: [], explicitNone: true };

  if (/^[\[{]/.test(body)) {
    try {
      const names = namesFromJson(JSON.parse(body)).map(cleanLLMMentionName).filter((x): x is string => !!x);
      return { names: dedupe(names), explicitNone: names.length === 0 };
    } catch {
      // Fall through to line parsing; some models produce near-JSON lists.
    }
  }

  const names = body.split("\n").map(cleanLLMMentionName).filter((x): x is string => !!x);
  return { names: dedupe(names), explicitNone: names.length === 0 && body.split("\n").every((l) => NULL_MENTION_RE.test(l.trim())) };
}

export class HeuristicExtractor implements Extractor {
  async extract(text: string): Promise<string[]> {
    const t = text.trim();
    if (!t) return [];
    const quoted = [...t.matchAll(QUOTED)].map((m) => m[1]!.trim());
    if (quoted.length) return dedupe(quoted);
    const candidates: string[] = [];
    for (const m of t.matchAll(PROPER)) {
      const span = cleanProperName(m[1]!.trim());
      if (span) candidates.push(span);
    }
    return dedupe(candidates);
  }
}

export class LLMExtractor implements Extractor {
  constructor(
    private runner: LLMRunner,
    private model = "gpt-4o-mini",
  ) {}
  async extract(text: string): Promise<string[]> {
    const prompt = buildMentionsPrompt(text);
    try {
      const content = await this.runner.run({ prompt, model: this.model });
      const parsed = parseLLMMentionResponse(content);
      if (parsed.names.length || parsed.explicitNone) return parsed.names;
      throw new Error("LLM mention extraction returned no parseable place list");
    } catch (err) {
      throw new Error(`LLM mention extraction failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

export function getExtractor(cfg: Config, runner: LLMRunner | null): Extractor {
  return runner ? new LLMExtractor(runner, cfg.models.extractor) : new HeuristicExtractor();
}

// ===========================================================================
// 2. Concept extraction — text → canonical concepts (the "coffee" in a search)
// ===========================================================================
const WORD = /[a-z0-9']+/g;

export function conceptsFromTags(tags: string[]): string[] {
  const out = new Set<string>();
  for (const t of tags) {
    const c = TAG_TO_CONCEPT[t.toLowerCase()];
    if (c) out.add(c);
  }
  return [...out];
}

export function extractConcepts(text: string): string[] {
  const lower = text.toLowerCase();
  const words = new Set(lower.match(WORD) ?? []);
  const found: string[] = [];
  const quietNoisePhrase = hasPhrase(lower, CONSTRAINT_MATCH_TERMS.noise.quiet.filter((t) => !t.startsWith("avoid ")));
  const lowCrowdPhrase = hasPhrase(lower, CONSTRAINT_MATCH_TERMS.crowd.low);
  for (const [concept, triggers] of Object.entries(CONCEPT_LEXICON)) {
    if (concept === "loud" && quietNoisePhrase) continue;
    if (concept === "crowded" && lowCrowdPhrase) continue;
    const hit = triggers.some((t) => (/[ -]/.test(t) ? lower.includes(t) : words.has(t)));
    if (hit) found.push(concept);
  }
  return found;
}

// ===========================================================================
// 3. Intent inference — fuzzy query → likely purposes (romantic → date)
// ===========================================================================
export interface ScoredIntent {
  purpose: string;
  score: number;
}

const COMPANION_PATTERNS: Array<[RegExp, string]> = [
  [/\b(kids?|children|child|toddler)\b/i, "kids"],
  [/\b(parents?|mom|dad|mother|father|grandma|grandpa)\b/i, "parents"],
  [/\b(family)\b/i, "family"],
  [/\b(partner|girlfriend|boyfriend|wife|husband|spouse|date night|my date)\b/i, "partner"],
  [/\b(client|colleague|coworker|co-worker|boss)\b/i, "client"],
  [/\b(friends?|buddies|group|crew)\b/i, "friends"],
  [/\b(alone|myself|solo|by myself)\b/i, "alone"],
];

/** Best-effort companion inference from free text (lexicon fallback). */
export function inferCompanion(text: string): string | null {
  for (const [re, who] of COMPANION_PATTERNS) if (re.test(text)) return who;
  return null;
}

const RELATIONSHIP_PATTERNS: Array<[RegExp, string]> = [
  [/\b(hated|terrible|awful|disliked|never again|worst|avoid)\b/i, "disliked"],
  [/\b(loved|amazing|favou?rite|best|obsessed)\b/i, "loved"],
  [/\b(want to (go|try|check)|wanna go|should try|on my list|need to (go|try)|planning to)\b/i, "want_to_go"],
  [/\b(went|visited|tried|been to|ate at|stopped by|grabbed|had (lunch|dinner|coffee))\b/i, "visited"],
  [/\b(liked|enjoyed|nice|solid|decent|pretty good)\b/i, "liked"],
];

/** Best-effort relationship inference from a sentence (lexicon fallback).
 * The LLM observe path does this per-mention; this is the key-free default. */
export function inferRelationship(text: string): string {
  for (const [re, rel] of RELATIONSHIP_PATTERNS) if (re.test(text)) return rel;
  return "mentioned";
}

export interface Measure {
  term: string;
  value: number;
}

/** Pull numeric measures (distance/walk-time/spend/ambient/travel friction) from a turn so an accepted
 * option teaches the calibration layer (e.g. agent says "3km away", user picks
 * it → learn near=3). Key-free; the LLM path can do this more robustly. */
export function extractMeasures(text: string): Measure[] {
  const out: Measure[] = [];
  const hasTerm = (term: string) => out.some((m) => m.term === term);
  const add = (term: string, value: number) => {
    if (Number.isFinite(value)) out.push({ term, value });
  };
  const addIfMissing = (term: string, value: number) => {
    if (!hasTerm(term)) add(term, value);
  };
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*km\b/gi)) add("near", Number(m[1]));
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*m(?:eters)?\b/gi)) add("near", Number(m[1]) / 1000);
  if (/\bwalk|on foot|步行/i.test(text))
    for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b/gi))
      add("walk_time", Number(m[1]));
  for (const m of text.matchAll(/[$¥€£]\s*(\d+(?:\.\d+)?)/g)) add("budget", Number(m[1]));
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*(?:yen|usd|dollars?|rmb|元|块)\b/gi))
    add("budget", Number(m[1]));

  const lower = text.toLowerCase();
  for (const m of lower.matchAll(/(?:noise|sound|ambient)\s*(?:level|score|rating)?\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\b/g))
    add("noise", Math.min(1, Number(m[1]) / 10));
  for (const m of lower.matchAll(/\b(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\s*(?:noise|sound|ambient)\b/g))
    add("noise", Math.min(1, Number(m[1]) / 10));
  for (const m of lower.matchAll(/\b(\d+(?:\.\d+)?)\s*db\b/g))
    add("noise", Math.max(0, Math.min(1, (Number(m[1]) - 35) / 50)));
  if (!hasTerm("noise")) {
    if (hasPhrase(lower, CONSTRAINT_MATCH_TERMS.noise.quiet))
      add("noise", 0.2);
    else if (/\b(?:moderate noise|some noise|background noise)\b/.test(lower))
      add("noise", 0.5);
    else if (hasPhrase(lower, CONSTRAINT_MATCH_TERMS.noise.loud) || /\b(?:deafening)\b/.test(lower))
      add("noise", 0.85);
  }

  for (const m of lower.matchAll(/(?:crowd|busy|busyness)\s*(?:level|score|rating)?\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\b/g))
    add("crowd", Math.min(1, Number(m[1]) / 10));
  for (const m of lower.matchAll(/\b(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\s*(?:crowd|busy|busyness)\b/g))
    add("crowd", Math.min(1, Number(m[1]) / 10));
  if (!hasTerm("crowd")) {
    if (hasPhrase(lower, CONSTRAINT_MATCH_TERMS.crowd.low))
      add("crowd", 0.2);
    else if (/\b(?:moderately crowded|some crowd|some line|some queue)\b/.test(lower))
      add("crowd", 0.5);
    else if (hasPhrase(lower, CONSTRAINT_MATCH_TERMS.crowd.high) || /\b(?:crowds?|long line|long queue)\b/.test(lower))
      add("crowd", 0.85);
  }

  const transit = `(?:${CONSTRAINT_MATCH_TERMS.travelMode.transit.map(termPattern).join("|")})`;
  for (const m of lower.matchAll(new RegExp(`(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes)\\s*(?:walk\\s*)?(?:from|to|away from|to the|from the)\\s+(?:the\\s+)?${transit}\\b`, "g")))
    add("transit_walk", Number(m[1]));
  for (const m of lower.matchAll(new RegExp(`${transit}\\s*(?:is\\s*)?(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes)\\s*(?:walk|away)\\b`, "g")))
    add("transit_walk", Number(m[1]));
  if (!hasTerm("transit_walk")) {
    if (new RegExp(`\\b(?:next to|beside|attached to|inside|at)\\s+(?:the\\s+)?${transit}\\b`).test(lower))
      add("transit_walk", 2);
    else if (new RegExp(`\\b(?:near|close to|by)\\s+(?:the\\s+)?${transit}\\b`).test(lower))
      add("transit_walk", 5);
  }
  return out;
}

function hasPhrase(lower: string, terms: string[]): boolean {
  return terms.some((term) => new RegExp(`\\b${termPattern(term)}\\b`).test(lower));
}

function termPattern(term: string): string {
  return term.split(/[\s_-]+/).map(escapeRegExp).join("[-_\\s]+");
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function inferIntents(text: string): ScoredIntent[] {
  const lower = text.toLowerCase();
  const words = new Set(lower.match(WORD) ?? []);
  const out: ScoredIntent[] = [];
  for (const [purpose, triggers] of Object.entries(INTENT_LEXICON)) {
    let score = 0;
    for (const t of triggers) if (t.includes(" ") ? lower.includes(t) : words.has(t)) score++;
    if (score > 0) out.push({ purpose, score });
  }
  out.sort((a, b) => b.score - a.score);
  return out;
}
