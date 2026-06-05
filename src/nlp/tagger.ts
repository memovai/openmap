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
  return f;
}

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
      f.constraints = {
        openNow: typeof c.openNow === "boolean" ? c.openNow : undefined,
        maxBudget: ["low", "mid", "high"].includes(c.maxBudget) ? c.maxBudget : undefined,
        dietary: arr(c.dietary),
        walkable: typeof c.walkable === "boolean" ? c.walkable : undefined,
      };
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
