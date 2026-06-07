import { type Config } from "../core/config.js";
import { type Relationship } from "../core/types.js";
import { type LLMRunner, extractJson } from "./llm.js";
import { type Measure, HeuristicExtractor, conceptsFromTags, extractConcepts, extractMeasures, inferCompanion, inferRelationship } from "./extract.js";
import { lexiconFrame } from "./tagger.js";
import { MEMORY_RELATIONSHIPS, buildMemoryPrompt } from "../prompts/memory.js";
import { MEASURE_TERMS, SCOPED_FALLBACK_MEASURE_TERMS } from "../core/vocabulary.js";

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
 * key-free pieces for explicit tests/eval only. */
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
    const scopeNames = [...new Set([...names, ...(await this.mentions.extract(full)).filter((m) => m.trim().toLowerCase() !== full.trim().toLowerCase())])];
    return names.map((name) => ({
      name,
      relationship: scopedRelationship(text, name, relationship, scopeNames),
      companions: companion ? [companion] : [],
      region: scopedRegion(full, text, name, scopeNames),
      measures: scopedMeasures(full, text, name, fullMeasures, scopeNames),
      goal,
      concepts: scopedConcepts(full, text, name, scopeNames),
    }));
  }
}

const CLAUSE_SPLIT = /\s*(?:,?\s+but\s+|;\s*|\.\s+|,\s+and\s+)\s*/i;
const REGION_RE = /\b(?:in|around|near)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})\b/;

function clauseForName(text: string, name: string, names: string[] = [name]): string {
  const lower = text.toLowerCase();
  const needle = name.toLowerCase();
  const idx = lower.indexOf(needle);
  if (idx < 0) return text.split(CLAUSE_SPLIT).find((c) => c.includes(name)) ?? text;

  let start = 0;
  for (const sep of [".", ";", "\n"]) {
    const at = text.lastIndexOf(sep, idx);
    if (at >= start) start = at + 1;
  }
  const commaBefore = text.lastIndexOf(",", idx);
  if (commaBefore >= start) start = commaBefore + 1;

  let end = text.length;
  for (const other of names) {
    if (other === name) continue;
    const otherIdx = lower.indexOf(other.toLowerCase(), idx + needle.length);
    if (otherIdx > idx && otherIdx < end) end = otherIdx;
  }
  for (const sep of [".", ";", "\n"]) {
    const at = findSeparator(text, sep, idx + needle.length);
    if (at >= 0 && at < end) end = at;
  }
  return text.slice(start, end).replace(/^[,\s]+|[,\s]+$/g, "") || text;
}

function findSeparator(text: string, sep: string, from: number): number {
  let at = text.indexOf(sep, from);
  while (at >= 0 && sep === "." && /\d/.test(text[at - 1] ?? "") && /\d/.test(text[at + 1] ?? ""))
    at = text.indexOf(sep, at + 1);
  return at;
}

function scopedRelationship(text: string, name: string, fallback: Relationship, names: string[]): Relationship {
  const rel = inferRelationship(clauseForName(text, name, names)) as Relationship;
  return rel === "mentioned" ? fallback : rel;
}

function scopedConcepts(full: string, text: string, name: string, names: string[]): string[] {
  const scoped = names.length <= 1 ? `${full} ${text}` : `${clauseForName(full, name, names)} ${clauseForName(text, name, names)}`;
  return [...new Set(extractConcepts(stripPlaceNames(scoped, names)))];
}

function scopedRegion(full: string, text: string, name: string, names: string[]): string | null {
  const specific = clauseForName(full, name, names).match(REGION_RE)?.[1];
  if (specific) return specific;
  return clauseForName(text, name, names).match(REGION_RE)?.[1] ?? null;
}

function scopedMeasures(full: string, text: string, name: string, fallback: Measure[], names: string[]): Measure[] {
  const scopedText = stripPlaceNames(`${clauseForName(full, name, names)} ${clauseForName(text, name, names)}`, names);
  const scoped = extractMeasures(scopedText);
  if (!scoped.length) return names.length <= 1 ? fallback : [];
  const out = [...scoped];
  const seen = new Set(out.map((m) => m.term));
  for (const term of SCOPED_FALLBACK_MEASURE_TERMS) {
    if (seen.has(term)) continue;
    const matches = fallback.filter((m) => m.term === term);
    if (matches.length !== 1) continue;
    out.push(matches[0]!);
    seen.add(term);
  }
  return out;
}

function stripPlaceNames(text: string, names: string[]): string {
  let out = text;
  for (const name of names) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(`\\b${escaped}\\b`, "gi"), " ");
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
      const parsedPlaces = places.filter((p: any) => p && typeof p.name === "string" && p.name.trim());
      const names = parsedPlaces.map((p: any) => String(p.name).trim());
      const full = `${opts.context ?? ""} ${text}`.trim();
      const out: ExtractedPlace[] = places
        .filter((p: any) => p && typeof p.name === "string" && p.name.trim())
        .map((p: any) => {
          const name = String(p.name).trim();
          const fb = fallbackByName.get(name.toLowerCase());
          const evidence = stripPlaceNames(`${clauseForName(full, name, names)} ${clauseForName(text, name, names)}`, names);
          const measures = mergeMeasures(
            sanitizeMeasures(Array.isArray(p.measures) ? p.measures : [], evidence),
            fb?.measures ?? [],
          );
          const concepts = normalizeConcepts(Array.isArray(p.concepts) ? p.concepts : [], evidence, fb?.concepts ?? []);
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
      return out;
    } catch (err) {
      throw new Error(`LLM memory extraction failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

const VALID_MEASURE_TERMS = new Set<string>(MEASURE_TERMS);
const AMBIENT_MEASURE_TERMS = new Set(["noise", "crowd"]);

function sanitizeMeasures(items: unknown[], evidenceText = ""): Measure[] {
  const evidenceByTerm = measuresByTerm(extractMeasures(evidenceText));
  return items
    .map(measureInput)
    .filter((m): m is MeasureInput => !!m && VALID_MEASURE_TERMS.has(m.term) && validMeasureValue(m) && hasMeasureEvidence(m, evidenceText, evidenceByTerm))
    .map((m) => normalizeMeasureFromEvidence(m, evidenceByTerm));
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

function measuresByTerm(measures: Measure[]): Map<string, Measure[]> {
  const out = new Map<string, Measure[]>();
  for (const m of measures) out.set(m.term, [...(out.get(m.term) ?? []), m]);
  return out;
}

function validMeasureValue(m: Measure): boolean {
  if (!Number.isFinite(m.value)) return false;
  if (m.term === "noise" || m.term === "crowd") return m.value >= 0 && m.value <= 1;
  return m.value > 0;
}

function normalizeAmbientMeasure(measure: Measure, evidence: Measure[]): Measure {
  if (!AMBIENT_MEASURE_TERMS.has(measure.term)) return measure;
  const grounded = evidence[0];
  if (!grounded) return measure;
  const lowCue = grounded.value <= 0.35;
  const highCue = grounded.value >= 0.65;
  if ((lowCue && measure.value > 0.35) || (highCue && measure.value < 0.65))
    return { term: measure.term, value: grounded.value };
  return measure;
}

interface MeasureInput extends Measure {
  evidence: string | null;
}

function measureInput(item: unknown): MeasureInput | null {
  if (!item || typeof item !== "object") return null;
  const raw = item as Record<string, unknown>;
  const evidence = raw.evidence ?? raw.evidenceText ?? raw.snippet ?? null;
  return {
    term: String(raw.term ?? "").toLowerCase().trim(),
    value: Number(raw.value),
    evidence: typeof evidence === "string" && evidence.trim() ? evidence.trim() : null,
  };
}

function hasMeasureEvidence(m: MeasureInput, evidenceText: string, evidenceByTerm: Map<string, Measure[]>): boolean {
  if (m.evidence && evidenceContains(evidenceText, m.evidence)) return true;
  return evidenceByTerm.has(m.term);
}

function normalizeMeasureFromEvidence(input: MeasureInput, evidenceByTerm: Map<string, Measure[]>): Measure {
  const measure = { term: input.term, value: input.value };
  const cited = input.evidence ? extractMeasures(input.evidence).filter((m) => m.term === input.term) : [];
  const grounded = cited.length ? cited : evidenceByTerm.get(input.term) ?? [];
  if (!AMBIENT_MEASURE_TERMS.has(input.term) && grounded.length) return grounded[0]!;
  return normalizeAmbientMeasure(measure, grounded);
}

function normalizeConcepts(items: unknown[], evidenceText = "", fallback: string[] = []): string[] {
  const out = new Set<string>();
  for (const concept of fallback) if (concept) out.add(concept);
  const evidenceConcepts = new Set(extractConcepts(evidenceText));
  const evidence = evidenceText.toLowerCase().replace(/[_-]+/g, " ");
  for (const item of items) {
    const input = conceptInput(item);
    if (!input) continue;
    const raw = input.tag;
    if (!raw || /^not[-_\s]?/.test(raw)) continue;
    const normalized = raw.replace(/[_-]+/g, " ");
    const mapped = [...conceptsFromTags([raw]), ...extractConcepts(normalized)];
    const grounded = (input.evidence && evidenceContains(evidenceText, input.evidence)) || evidence.includes(normalized) || mapped.some((m) => evidenceConcepts.has(m));
    if (!grounded) continue;
    if (mapped.length) {
      for (const concept of mapped) out.add(concept);
    } else {
      out.add(raw.replace(/\s+/g, "_"));
    }
  }
  if (out.has("low_crowd")) out.delete("crowded");
  if (out.has("quiet")) out.delete("loud");
  return [...out];
}

interface ConceptInput {
  tag: string;
  evidence: string | null;
}

function conceptInput(item: unknown): ConceptInput | null {
  if (typeof item === "string") return { tag: item.toLowerCase().trim(), evidence: null };
  if (!item || typeof item !== "object") return null;
  const raw = item as Record<string, unknown>;
  const tag = raw.tag ?? raw.concept ?? raw.name ?? raw.value;
  const evidence = raw.evidence ?? raw.evidenceText ?? raw.snippet ?? null;
  if (typeof tag !== "string") return null;
  return {
    tag: tag.toLowerCase().trim(),
    evidence: typeof evidence === "string" && evidence.trim() ? evidence.trim() : null,
  };
}

function evidenceContains(text: string, phrase: string): boolean {
  const haystack = normalizeEvidence(text);
  const needle = normalizeEvidence(phrase);
  return needle.length >= 2 && haystack.includes(needle);
}

function normalizeEvidence(text: string): string {
  return text
    .toLowerCase()
    .replace(/[“”"‘']/g, "")
    .replace(/[_\-\s]+/g, " ")
    .trim();
}

export function getMemoryExtractor(cfg: Config, runner: LLMRunner | null): MemoryExtractor {
  return runner ? new LLMMemoryExtractor(runner, cfg.models.extractor) : new HeuristicMemoryExtractor();
}
