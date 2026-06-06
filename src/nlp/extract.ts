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
  open_late: ["open late", "late-night", "late night", "open now", "24/7", "after midnight"],
  walkable: ["walkable", "walk", "walking distance", "on foot"],
  low_crowd: ["uncrowded", "not crowded", "not busy", "no crowds", "no crowd", "no line", "no queue"],
  crowded: ["crowded", "busy", "packed", "crowd", "crowds", "line", "queue"],
  transit: ["transit", "station", "subway", "metro", "train", "bus"],
  parking: ["parking", "drive", "driving", "car", "valet"],
  // vibe concepts
  cozy: ["cozy", "cosy", "intimate", "snug"],
  quiet: ["quiet", "calm", "peaceful", "low noise", "low-noise"],
  loud: ["loud", "noisy", "noise", "too noisy", "loud music"],
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
  cozy: "cozy",
  quiet: "quiet",
  calm: "quiet",
  loud: "loud",
  noisy: "loud",
  romantic: "romantic",
  lively: "lively",
  outdoor: "outdoor",
  patio: "outdoor",
  terrace: "outdoor",
  vegan: "vegetarian",
  vegetarian: "vegetarian",
  plant_based: "vegetarian",
  open_late: "open_late",
  late_night: "open_late",
  walkable: "walkable",
  low_crowd: "low_crowd",
  uncrowded: "low_crowd",
  crowded: "crowded",
  busy: "crowded",
  transit: "transit",
  station: "transit",
  subway: "transit",
  metro: "transit",
  train: "transit",
  bus: "transit",
  parking: "parking",
  valet: "parking",
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
  const quietNoisePhrase = /\b(?:low[- ]noise|not noisy|not loud|not too noisy|not too loud|without noise|no loud music)\b/.test(lower);
  for (const [concept, triggers] of Object.entries(CONCEPT_LEXICON)) {
    if (concept === "loud" && quietNoisePhrase) continue;
    const hit = triggers.some((t) => (/[ -]/.test(t) ? lower.includes(t) : words.has(t)));
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
  study: ["study", "studying", "library"],
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
export const VIBE_CONCEPTS = new Set(["cozy", "quiet", "loud", "romantic", "lively", "outdoor", "cheap", "fancy"]);

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
    if (/\b(?:quiet|calm|peaceful|low[- ]noise|not noisy|not loud|not too noisy|not too loud|without noise|no loud music)\b/.test(lower))
      add("noise", 0.2);
    else if (/\b(?:moderate noise|some noise|background noise)\b/.test(lower))
      add("noise", 0.5);
    else if (/\b(?:loud|noisy|too noisy|loud music|deafening)\b/.test(lower))
      add("noise", 0.85);
  }

  for (const m of lower.matchAll(/(?:crowd|busy|busyness)\s*(?:level|score|rating)?\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\b/g))
    add("crowd", Math.min(1, Number(m[1]) / 10));
  for (const m of lower.matchAll(/\b(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)\s*(?:crowd|busy|busyness)\b/g))
    add("crowd", Math.min(1, Number(m[1]) / 10));
  if (!hasTerm("crowd")) {
    if (/\b(?:uncrowded|not crowded|not busy|not too crowded|not too busy|without crowds?|no crowds?|no line|no queue)\b/.test(lower))
      add("crowd", 0.2);
    else if (/\b(?:moderately crowded|some crowd|some line|some queue)\b/.test(lower))
      add("crowd", 0.5);
    else if (/\b(?:crowded|busy|packed|crowds?|long line|long queue)\b/.test(lower))
      add("crowd", 0.85);
  }

  const transit = "(?:station|subway|metro|train|bus|transit)";
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
