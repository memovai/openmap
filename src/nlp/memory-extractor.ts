import { type Config } from "../core/config.js";
import { type Relationship } from "../core/types.js";
import { type LLMRunner, extractJson } from "./llm.js";
import { type Measure, HeuristicExtractor, extractConcepts, extractMeasures, inferCompanion, inferRelationship } from "./extract.js";
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
    const full = `${opts.context ?? ""} ${text}`.trim();
    const fullMeasures = extractMeasures(full);
    return names.map((name) => ({
      name,
      relationship: scopedRelationship(text, name, relationship),
      companions: companion ? [companion] : [],
      region: scopedRegion(full, text, name),
      measures: scopedMeasures(full, text, name, fullMeasures),
      goal,
      concepts: scopedConcepts(full, text, name),
    }));
  }
}

const CLAUSE_SPLIT = /\s*(?:,?\s+but\s+|;\s*|\.\s+|,\s+and\s+)\s*/i;
const REGION_RE = /\b(?:in|around|near)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})\b/;

function clauseForName(text: string, name: string): string {
  return text.split(CLAUSE_SPLIT).find((c) => c.includes(name)) ?? text;
}

function scopedRelationship(text: string, name: string, fallback: Relationship): Relationship {
  const rel = inferRelationship(clauseForName(text, name)) as Relationship;
  return rel === "mentioned" ? fallback : rel;
}

function scopedConcepts(full: string, text: string, name: string): string[] {
  return [...new Set([...extractConcepts(clauseForName(full, name)), ...extractConcepts(clauseForName(text, name))])];
}

function scopedRegion(full: string, text: string, name: string): string | null {
  const specific = clauseForName(full, name).match(REGION_RE)?.[1];
  if (specific) return specific;
  return clauseForName(text, name).match(REGION_RE)?.[1] ?? null;
}

function scopedMeasures(full: string, text: string, name: string, fallback: Measure[]): Measure[] {
  const scoped = extractMeasures(`${clauseForName(full, name)} ${clauseForName(text, name)}`);
  if (!scoped.length) return fallback;
  const out = [...scoped];
  const seen = new Set(out.map((m) => m.term));
  for (const term of ["budget", "walk_time", "transit_walk"]) {
    if (seen.has(term)) continue;
    const matches = fallback.filter((m) => m.term === term);
    if (matches.length !== 1) continue;
    out.push(matches[0]!);
    seen.add(term);
  }
  return out;
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
      const fallback = await this.fallback.extract(text, opts);
      const fallbackByName = new Map(fallback.map((p) => [p.name.toLowerCase(), p]));
      const out: ExtractedPlace[] = places
        .filter((p: any) => p && typeof p.name === "string" && p.name.trim())
        .map((p: any) => {
          const name = String(p.name).trim();
          const fb = fallbackByName.get(name.toLowerCase());
          const measures = mergeMeasures(
            sanitizeMeasures(Array.isArray(p.measures) ? p.measures : []),
            fb?.measures ?? [],
          );
          const concepts = [
            ...new Set([
              ...sanitizeConcepts(Array.isArray(p.concepts) ? p.concepts : []),
              ...(fb?.concepts ?? []),
            ]),
          ];
          return {
            name,
            relationship: (MEMORY_RELATIONSHIPS.includes(p.relationship) ? p.relationship : fb?.relationship ?? "mentioned") as Relationship,
            companions: Array.isArray(p.companions) ? p.companions.map(String) : fb?.companions ?? [],
            region: p.region ? String(p.region) : fb?.region ?? null,
            measures,
            goal: p.goal ? String(p.goal) : fb?.goal ?? null,
            concepts,
          };
        });
      return out.length ? out : this.fallback.extract(text, opts);
    } catch {
      return this.fallback.extract(text, opts);
    }
  }
}

const VALID_MEASURE_TERMS = new Set(["near", "walk_time", "budget", "noise", "crowd", "transit_walk"]);

function sanitizeMeasures(items: unknown[]): Measure[] {
  return items
    .map((m: any) => ({ term: String(m?.term ?? "").toLowerCase().trim(), value: Number(m?.value) }))
    .filter((m: Measure) => VALID_MEASURE_TERMS.has(m.term) && Number.isFinite(m.value) && (m.term === "noise" || m.term === "crowd" ? m.value >= 0 : m.value > 0));
}

function mergeMeasures(primary: Measure[], fallback: Measure[]): Measure[] {
  const out = [...primary];
  const seen = new Set(out.map((m) => m.term));
  for (const m of fallback) {
    if (!VALID_MEASURE_TERMS.has(m.term) || seen.has(m.term)) continue;
    out.push(m);
    seen.add(m.term);
  }
  return out;
}

function sanitizeConcepts(items: unknown[]): string[] {
  return items
    .map((c: unknown) => String(c).toLowerCase().trim())
    .filter((c: string) => c && !/^not[-_\s]?/.test(c));
}

export function getMemoryExtractor(cfg: Config, runner: LLMRunner | null): MemoryExtractor {
  return runner ? new LLMMemoryExtractor(runner, cfg.models.extractor) : new HeuristicMemoryExtractor();
}
