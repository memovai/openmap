import { type Config } from "../core/config.js";
import { type LLMRunner } from "./llm.js";
import { buildMentionsPrompt } from "../prompts/mentions.js";

// ===========================================================================
// 1. Mention extraction — free text → candidate place names (for resolution)
// ===========================================================================
export interface Extractor {
  extract(text: string): Promise<string[]>;
}

// Double-style quotes only — a straight apostrophe here would false-match
// contractions (e.g. "let's do X" → "s do X").
const QUOTED = /["“”]([^"“”]{2,60})["“”]/g;
const PROPER = /\b([A-Z][\w&'.-]+(?:\s+(?:[A-Z][\w&'.-]+|de|la|le|du|of|the|and|&)){0,4})/g;
const STOP_LEADING = new Set(["I", "We", "They", "My", "The", "A", "An", "Last", "This", "That"]);

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

export class HeuristicExtractor implements Extractor {
  async extract(text: string): Promise<string[]> {
    const t = text.trim();
    if (!t) return [];
    const quoted = [...t.matchAll(QUOTED)].map((m) => m[1]!.trim());
    if (quoted.length) return dedupe(quoted);
    const candidates: string[] = [];
    for (const m of t.matchAll(PROPER)) {
      const span = m[1]!.trim();
      const words = span.split(/\s+/);
      if (words.length === 1 && STOP_LEADING.has(words[0]!)) continue;
      candidates.push(span);
    }
    const deduped = dedupe(candidates);
    return deduped.length ? deduped : [t];
  }
}

export class LLMExtractor implements Extractor {
  private fallback = new HeuristicExtractor();
  constructor(
    private runner: LLMRunner,
    private model = "gpt-4o-mini",
  ) {}
  async extract(text: string): Promise<string[]> {
    const prompt = buildMentionsPrompt(text);
    try {
      const content = await this.runner.run({ prompt, model: this.model });
      const names = content.split("\n").map((l: string) => l.replace(/^[-•\s\t]+/, "").trim()).filter(Boolean);
      const d = dedupe(names);
      return d.length ? d : this.fallback.extract(text);
    } catch {
      return this.fallback.extract(text);
    }
  }
}

export function getExtractor(cfg: Config, runner: LLMRunner | null): Extractor {
  return runner ? new LLMExtractor(runner, cfg.models.extractor) : new HeuristicExtractor();
}

// ===========================================================================
// 2. Concept extraction — text → canonical concepts (the "coffee" in a search)
// ===========================================================================
// Each concept maps to trigger words/phrases. Single words match on word
// boundaries; multi-word phrases match as substrings.
const CONCEPT_LEXICON: Record<string, string[]> = {
  coffee: ["coffee", "cafe", "café", "espresso", "latte", "cappuccino", "flat white"],
  tea: ["tea", "matcha", "boba", "bubble tea"],
  ramen: ["ramen", "noodle", "noodles"],
  sushi: ["sushi", "sashimi", "omakase"],
  pizza: ["pizza", "pizzeria"],
  burger: ["burger", "burgers"],
  bar: ["bar", "pub", "cocktail", "cocktails", "beer", "brewery"],
  wine: ["wine", "wine bar", "winery"],
  brunch: ["brunch", "breakfast"],
  dessert: ["dessert", "cake", "bakery", "pastry", "ice cream", "gelato"],
  bbq: ["bbq", "barbecue", "grill"],
  vegetarian: ["vegetarian", "vegan", "plant-based"],
  // vibe concepts
  cozy: ["cozy", "cosy", "intimate", "snug"],
  quiet: ["quiet", "calm", "peaceful"],
  romantic: ["romantic", "candlelit"],
  lively: ["lively", "bustling", "vibrant"],
  outdoor: ["outdoor", "patio", "terrace", "rooftop", "garden seating"],
  cheap: ["cheap", "budget", "affordable"],
  fancy: ["fancy", "upscale", "fine dining", "michelin"],
};

const WORD = /[a-z0-9']+/g;

// Map provider tags (e.g. OSM `cafe`, `coffee_shop`) onto canonical concepts so
// behavioral signals from text and from resolved places share a vocabulary.
const TAG_TO_CONCEPT: Record<string, string> = {
  cafe: "coffee",
  coffee_shop: "coffee",
  coffee: "coffee",
  tea: "tea",
  bar: "bar",
  pub: "bar",
  biergarten: "bar",
  bakery: "dessert",
  pastry: "dessert",
  ice_cream: "dessert",
  confectionery: "dessert",
  wine: "wine",
  ramen: "ramen",
  sushi: "sushi",
  pizza: "pizza",
};

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
  for (const [concept, triggers] of Object.entries(CONCEPT_LEXICON)) {
    const hit = triggers.some((t) => (t.includes(" ") ? lower.includes(t) : words.has(t)));
    if (hit) found.push(concept);
  }
  return found;
}

// ===========================================================================
// 3. Intent inference — fuzzy query → likely purposes (romantic → date)
// ===========================================================================
const INTENT_LEXICON: Record<string, string[]> = {
  date: ["romantic", "intimate", "candlelit", "date", "date night", "couple"],
  romance: ["romantic", "anniversary", "candlelit"],
  work: ["work", "laptop", "wifi", "wi-fi", "remote", "co-working", "coworking"],
  study: ["study", "studying", "quiet", "library"],
  family: ["family", "kid", "kids", "children", "child-friendly", "family-friendly"],
  celebration: ["birthday", "celebrate", "celebration", "party", "anniversary"],
  business: ["business", "client", "meeting", "professional"],
  hangout: ["friends", "group", "hang out", "hangout", "casual", "catch up"],
  solo: ["solo", "alone", "by myself"],
};

export interface ScoredIntent {
  purpose: string;
  score: number;
}

/** Which extracted concepts are "vibe/affordance" (how it feels) vs category. */
export const VIBE_CONCEPTS = new Set(["cozy", "quiet", "romantic", "lively", "outdoor", "cheap", "fancy"]);

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

/** Pull numeric measures (distance/walk-time/spend) from a turn so an accepted
 * option teaches the calibration layer (e.g. agent says "3km away", user picks
 * it → learn near=3). Key-free; the LLM path can do this more robustly. */
export function extractMeasures(text: string): Measure[] {
  const out: Measure[] = [];
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*km\b/gi)) out.push({ term: "near", value: Number(m[1]) });
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*m(?:eters)?\b/gi)) out.push({ term: "near", value: Number(m[1]) / 1000 });
  if (/\bwalk|on foot|步行/i.test(text))
    for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b/gi))
      out.push({ term: "walk_time", value: Number(m[1]) });
  for (const m of text.matchAll(/[$¥€£]\s*(\d+(?:\.\d+)?)/g)) out.push({ term: "budget", value: Number(m[1]) });
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*(?:yen|usd|dollars?|rmb|元|块)\b/gi))
    out.push({ term: "budget", value: Number(m[1]) });
  return out;
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
