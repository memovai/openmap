import { type Config } from "../core/config.js";
import { type Relationship } from "../core/types.js";
import { type LLMRunner, extractJson } from "./llm.js";
import { type Measure, HeuristicExtractor, extractMeasures, inferCompanion, inferRelationship } from "./extract.js";
import { lexiconFrame } from "./tagger.js";
import { MEMORY_RELATIONSHIPS, buildMemoryPrompt } from "../prompts/memory.js";

/** A structured place extraction from one conversational turn. */
export interface ExtractedPlace {
  name: string;
  relationship: Relationship; // per-mention sentiment (not one label for the whole sentence)
  companions: string[];
  region: string | null; // a named area the place is in ("Shibuya"), if stated
  measures: Measure[]; // distance/walk/spend the place was described with
  goal: string | null; // the turn's intent context (for context-conditioned calibration)
  concepts?: string[]; // category/vibe tags for THIS place (e.g. "quiet coffee spot" → [coffee, quiet])
}

/** Turns conversation into structured place memories. The LLM version does
 * per-mention sentiment, cross-sentence linkage, named regions, and attaches
 * the right measure to the right place; the heuristic version composes the
 * key-free pieces and preserves offline behavior. */
export interface MemoryExtractor {
  extract(text: string, opts?: { context?: string }): Promise<ExtractedPlace[]>;
}

export class HeuristicMemoryExtractor implements MemoryExtractor {
  private mentions = new HeuristicExtractor();

  async extract(text: string, opts: { context?: string } = {}): Promise<ExtractedPlace[]> {
    const names = (await this.mentions.extract(text)).filter(
      (m) => m.trim().toLowerCase() !== text.trim().toLowerCase(), // drop the whole-sentence fallback
    );
    const relationship = inferRelationship(text) as Relationship;
    const companion = inferCompanion(text);
    const goal = lexiconFrame(text).goals[0] ?? null;
    const measures = extractMeasures(`${opts.context ?? ""} ${text}`);
    return names.map((name) => ({
      name,
      relationship,
      companions: companion ? [companion] : [],
      region: null,
      measures,
      goal,
    }));
  }
}

export class LLMMemoryExtractor implements MemoryExtractor {
  constructor(
    private runner: LLMRunner,
    private model: string,
    private fallback: MemoryExtractor = new HeuristicMemoryExtractor(),
  ) {}

  async extract(text: string, opts: { context?: string } = {}): Promise<ExtractedPlace[]> {
    const prompt = buildMemoryPrompt(text, opts.context ?? "");
    try {
      const content = await this.runner.run({ prompt, json: true, model: this.model });
      const parsed = JSON.parse(extractJson(content || "{}"));
      const places = Array.isArray(parsed.places) ? parsed.places : [];
      const out: ExtractedPlace[] = places
        .filter((p: any) => p && typeof p.name === "string" && p.name.trim())
        .map((p: any) => ({
          name: String(p.name).trim(),
          relationship: (MEMORY_RELATIONSHIPS.includes(p.relationship) ? p.relationship : "mentioned") as Relationship,
          companions: Array.isArray(p.companions) ? p.companions.map(String) : [],
          region: p.region ? String(p.region) : null,
          measures: Array.isArray(p.measures)
            ? p.measures
                .filter((m: any) => ["near", "walk_time", "budget"].includes(m?.term) && Number.isFinite(Number(m?.value)))
                .map((m: any) => ({ term: String(m.term), value: Number(m.value) }))
            : [],
          goal: p.goal ? String(p.goal) : null,
          concepts: Array.isArray(p.concepts) ? p.concepts.map((c: unknown) => String(c).toLowerCase().trim()).filter(Boolean) : [],
        }));
      return out.length ? out : this.fallback.extract(text, opts);
    } catch {
      return this.fallback.extract(text, opts);
    }
  }
}

export function getMemoryExtractor(cfg: Config, runner: LLMRunner | null): MemoryExtractor {
  return runner ? new LLMMemoryExtractor(runner, cfg.models.extractor) : new HeuristicMemoryExtractor();
}
