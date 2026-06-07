import { type DB } from "../store/db.js";
import { type MeasureTerm } from "../core/vocabulary.js";

/**
 * Calibration layer — the user's *personal meaning* of fuzzy place predicates,
 * learned from behavior. "near" is just one term: many words in a maps query
 * ("walkable", "cheap", "quiet"…) are subjective and per-user. Rather than
 * hardcode each, we learn a numeric value per term from accepted/confirmed
 * evidence, and resolve queries against it.
 *
 * Aggregation per term:
 *  - "max": a tolerance ceiling — the largest accepted value (near, walk_time,
 *    noise, crowd, transit_walk).
 *  - "min": a floor — the smallest accepted value.
 *  - "ema": a running typical value (budget, noise level).
 */
export type Agg = "max" | "min" | "ema";

export interface TermSpec {
  defaultValue: number | null; // prior before any learning (null = unknown)
  agg: Agg;
  unit: string;
  note: string;
}

export const TERMS: Record<MeasureTerm, TermSpec> = {
  near: { defaultValue: 2, agg: "max", unit: "km", note: "radius the user accepts as 'near'" },
  walk_time: { defaultValue: 10, agg: "max", unit: "min", note: "how long they'll walk" },
  budget: { defaultValue: null, agg: "ema", unit: "ccy", note: "typical accepted spend" },
  noise: { defaultValue: 0.35, agg: "max", unit: "0..1", note: "highest accepted ambient noise level" },
  crowd: { defaultValue: 0.35, agg: "max", unit: "0..1", note: "highest accepted crowd level" },
  transit_walk: { defaultValue: 8, agg: "max", unit: "min", note: "walk time from transit the user accepts" },
};

const specFor = (term: string): TermSpec | undefined => TERMS[term as MeasureTerm];

export interface Calibration {
  term: string;
  value: number | null;
  samples: number;
  unit: string;
}

const keyOf = (term: string, context?: string) => (context ? `${term}@${context}` : term);
const baseTerm = (key: string) => key.split("@")[0]!;

/**
 * Resolve a term, optionally for a context (e.g. goal "date"). A fuzzy word
 * means different things in different situations — "near for a date" may be 5km
 * while "near for coffee" is 1km. Resolution prefers the context-specific value
 * (if learned), else falls back to the global value, else the prior.
 */
export function getCalibration(db: DB, userId: string, term: string, context?: string): Calibration {
  const spec = specFor(term);
  if (context) {
    const row = db.getCalibration(userId, keyOf(term, context));
    if (row && row.value != null && row.samples > 0)
      return { term: keyOf(term, context), value: row.value, samples: row.samples, unit: spec?.unit ?? "" };
  }
  const row = db.getCalibration(userId, term);
  return { term, value: row?.value ?? spec?.defaultValue ?? null, samples: row?.samples ?? 0, unit: spec?.unit ?? "" };
}

/** Update a term from one accepted sample (revealed preference). Always updates
 * the global value (the fallback); when a context is given, also the
 * context-specific value. */
export function learnCalibration(db: DB, userId: string, term: string, sample: number, context?: string): void {
  const spec = specFor(term);
  if (!spec || !Number.isFinite(sample)) return;
  const apply = (key: string) => {
    const cur = db.getCalibration(userId, key);
    let next: number;
    if (!cur || cur.value == null) next = sample;
    else if (spec.agg === "max") next = Math.max(cur.value, sample);
    else if (spec.agg === "min") next = Math.min(cur.value, sample);
    else next = cur.value * 0.7 + sample * 0.3; // ema
    db.upsertCalibration(userId, key, next, (cur?.samples ?? 0) + 1);
  };
  apply(term);
  if (context) apply(keyOf(term, context));
}

export function allCalibrations(db: DB, userId: string): Calibration[] {
  const base = (Object.keys(TERMS) as MeasureTerm[]).map((t) => getCalibration(db, userId, t));
  const scoped = db
    .listCalibrations(userId)
    .filter((r) => r.term.includes("@"))
    .map((r) => ({ term: r.term, value: r.value, samples: r.samples, unit: specFor(baseTerm(r.term))?.unit ?? "" }));
  return [...base, ...scoped];
}

/** Convenience: the learned meaning of "near" (km), optionally for a context. */
export function nearRadiusKm(db: DB, userId: string, context?: string): number {
  return getCalibration(db, userId, "near", context).value ?? (TERMS.near!.defaultValue as number);
}
