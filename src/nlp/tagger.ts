import { type Config } from "../core/config.js";
import { type LLMRunner, extractJson } from "./llm.js";
import { ALLOWED_GOALS, buildIntentPrompt } from "../prompts/intent.js";
import {
  type ScoredIntent,
  VIBE_CONCEPTS,
  extractConcepts,
  inferCompanion,
  inferIntents,
} from "./extract.js";
import { type IntentFrame, emptyFrame } from "../core/types.js";

/** Resolves the latent situational intent behind a maps query, plus the
 * concepts/intents used for behavioral memory. The lexicon version is key-free
 * and deterministic; the LLM version handles indirect phrasing
 * ("a spot to impress a first date" → goal date, vibe intimate). */
export interface Tagger {
  concepts(text: string): Promise<string[]>;
  intents(text: string): Promise<ScoredIntent[]>;
  frame(text: string): Promise<IntentFrame>;
}

/** Build an IntentFrame from the key-free lexicon. Handles simple negation
 * ("nothing too fancy") and companion/goal contradictions; the LLM tagger does
 * this properly — this just keeps the fallback from actively mis-reading. */
export function lexiconFrame(text: string): IntentFrame {
  const lower = text.toLowerCase();
  const negated = new Set(
    [...lower.matchAll(/\b(?:no|not|nothing(?: too)?|without|avoid|don'?t want)\s+([a-z]+)/g)].map((m) => m[1]!),
  );
  const concepts = extractConcepts(text).filter((c) => !negated.has(c));
  const f = emptyFrame(text);
  f.companions = inferCompanion(text);
  f.concepts = concepts.filter((c) => !VIBE_CONCEPTS.has(c));
  f.vibe = concepts.filter((c) => VIBE_CONCEPTS.has(c));
  f.goals = inferIntents(text).map((i) => i.purpose);
  // you don't take a "date" with your parents/kids — drop the contradiction
  if (f.companions && ["parents", "kids", "family"].includes(f.companions))
    f.goals = f.goals.filter((g) => g !== "date" && g !== "romance");
  if (concepts.includes("cheap")) f.constraints.maxBudget = "low";
  if (concepts.includes("fancy")) f.constraints.maxBudget = "high";
  if (concepts.includes("vegetarian")) f.constraints.dietary = ["vegetarian"];
  if (concepts.includes("open_late")) f.constraints.openNow = true;
  if (concepts.includes("walkable")) {
    f.constraints.walkable = true;
    f.constraints.travelMode = "walk";
  }
  if (wantsQuiet(lower)) f.constraints.noise = "quiet";
  else if (concepts.includes("loud")) f.constraints.noise = "loud";
  if (wantsLowCrowd(lower)) f.constraints.crowd = "low";
  else if (concepts.includes("crowded")) f.constraints.crowd = "high";
  if (/\b(?:transit|station|subway|metro|train|bus)\b/.test(lower)) f.constraints.travelMode = "transit";
  if (/\b(?:parking|drive|driving|car|valet)\b/.test(lower)) f.constraints.travelMode = "drive";
  return f;
}

const wantsQuiet = (lower: string) =>
  /\b(?:quiet|calm|peaceful|low noise|not noisy|not loud|not too noisy|not too loud|without noise|avoid noise|avoid noisy|no loud music)\b/.test(lower);

const wantsLowCrowd = (lower: string) =>
  /\b(?:uncrowded|not crowded|not busy|not too crowded|not too busy|without crowds?|avoid crowds?|no crowds?|no line|no queue)\b/.test(lower);

export class LexiconTagger implements Tagger {
  async frame(text: string): Promise<IntentFrame> {
    return lexiconFrame(text);
  }
  async concepts(text: string): Promise<string[]> {
    const f = lexiconFrame(text);
    return [...f.concepts, ...f.vibe];
  }
  async intents(text: string): Promise<ScoredIntent[]> {
    return inferIntents(text);
  }
}

export class LLMTagger implements Tagger {
  private fallback = new LexiconTagger();
  private memo: { text: string; frame: IntentFrame } | null = null;

  constructor(
    private runner: LLMRunner,
    private model = "gpt-4o-mini",
  ) {}

  /** One LLM call resolves the whole frame; concepts()/intents() derive from it. */
  async frame(text: string): Promise<IntentFrame> {
    if (this.memo && this.memo.text === text) return this.memo.frame;
    const prompt = buildIntentPrompt(text);
    try {
      const content = await this.runner.run({ prompt, json: true, model: this.model });
      const p = JSON.parse(extractJson(content || "{}"));
      const arr = (x: unknown): string[] =>
        Array.isArray(x) ? x.map((v) => String(v).toLowerCase().trim()).filter(Boolean) : [];
      const f = emptyFrame(text);
      f.goals = arr(p.goals).filter((g) => ALLOWED_GOALS.includes(g));
      f.companions = p.companions ? String(p.companions).toLowerCase() : null;
      f.occasion = p.occasion ? String(p.occasion) : null;
      f.concepts = arr(p.concepts);
      f.vibe = arr(p.vibe);
      const c = p.constraints ?? {};
      const choice = <T extends string>(x: unknown, allowed: readonly T[]): T | undefined => {
        const v = x == null ? "" : String(x).toLowerCase().trim();
        return (allowed as readonly string[]).includes(v) ? (v as T) : undefined;
      };
      f.constraints = {
        openNow: typeof c.openNow === "boolean" ? c.openNow : undefined,
        maxBudget: choice(c.maxBudget, ["low", "mid", "high"] as const),
        dietary: arr(c.dietary),
        walkable: typeof c.walkable === "boolean" ? c.walkable : undefined,
        noise: choice(c.noise, ["quiet", "moderate", "loud"] as const),
        crowd: choice(c.crowd, ["low", "moderate", "high"] as const),
        travelMode: choice(c.travelMode, ["walk", "transit", "drive"] as const),
      };
      const lexical = lexiconFrame(text);
      f.goals = [...new Set([...f.goals, ...lexical.goals])].filter((g) => ALLOWED_GOALS.includes(g));
      f.concepts = [...new Set([...f.concepts, ...lexical.concepts])];
      f.vibe = [...new Set([...f.vibe, ...lexical.vibe])];
      const merged = { ...lexical.constraints };
      for (const [k, v] of Object.entries(f.constraints)) {
        if (v !== undefined) (merged as Record<string, unknown>)[k] = v;
      }
      merged.dietary = [...new Set([...(lexical.constraints.dietary ?? []), ...(f.constraints.dietary ?? [])])];
      f.constraints = merged;
      if (f.companions && ["parents", "kids", "family"].includes(f.companions))
        f.goals = f.goals.filter((g) => g !== "date" && g !== "romance");
      this.memo = { text, frame: f };
      return f;
    } catch {
      return this.fallback.frame(text);
    }
  }

  async concepts(text: string): Promise<string[]> {
    const f = await this.frame(text);
    return [...new Set([...f.concepts, ...f.vibe])];
  }
  async intents(text: string): Promise<ScoredIntent[]> {
    return (await this.frame(text)).goals.map((purpose) => ({ purpose, score: 1 }));
  }
}

export function getTagger(cfg: Config, runner: LLMRunner | null): Tagger {
  if (runner && cfg.tagger !== "lexicon") return new LLMTagger(runner, cfg.models.tagger);
  return new LexiconTagger();
}
