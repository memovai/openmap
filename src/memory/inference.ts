import { type DB } from "../store/db.js";
import { conceptsFromTags, extractConcepts } from "../nlp/extract.js";
import { type Belief, type Memory, type Predicate, type Relationship, nowIso } from "../core/types.js";

// ── reconcile: decide ADD / UPDATE / NOOP for a new place observation ───────
export type ReconcileAction = "add" | "update" | "noop";
export interface ReconcileDecision {
  action: ReconcileAction;
  targetId: number | null;
  reason: string;
}

const ACTUAL = new Set<Relationship>(["visited", "loved", "liked", "disliked"]);

/** Reconcile a new (place, relationship) observation against what the user
 * already remembers about that place — the ADD/UPDATE/NOOP reconciliation step.
 * Latest actual experience wins over a prior plan or weaker sentiment. */
export function reconcileDecision(existing: Memory[], rel: Relationship): ReconcileDecision {
  if (existing.length === 0) return { action: "add", targetId: null, reason: "new place" };
  if (existing.some((m) => m.relationship === rel))
    return { action: "noop", targetId: null, reason: "already known" };

  // planned → actual: "want_to_go" becomes the real experience
  const want = existing.find((m) => m.relationship === "want_to_go");
  if (want && ACTUAL.has(rel)) return { action: "update", targetId: want.id ?? null, reason: "visited a planned place" };

  const strongest = existing.reduce((a, b) => (b.affect > a.affect ? b : a));
  // contradiction: latest sentiment wins
  if (rel === "disliked" && strongest.affect > 0)
    return { action: "update", targetId: strongest.id ?? null, reason: "sentiment flipped negative" };
  // richer sentiment after a plain visit
  if ((rel === "loved" || rel === "liked") && strongest.relationship === "visited")
    return { action: "update", targetId: strongest.id ?? null, reason: "sentiment upgraded" };
  // a bare new mention of a known place adds nothing
  if (rel === "mentioned") return { action: "noop", targetId: null, reason: "re-mention" };
  return { action: "add", targetId: null, reason: "new distinct relationship" };
}


// Confidence saturates with accumulated support: conf = 1 − e^(−K·signal).
const K = 0.6;
const EVENT_WEIGHT = 0.5; // each behavioral event nudges a preference
const LIKELY = 0.4; // threshold for "you probably like this"
const CONSOLIDATE_MIN = 0.3; // only persist beliefs above this

const round = (x: number) => Number(x.toFixed(3));

/** Recency decay for confidence. Inferred beliefs lose half their confidence
 * every `halfLifeDays` unless reinforced; stated beliefs never decay. */
export function decayConfidence(
  confidence: number,
  updatedAt: string,
  source: "inferred" | "stated",
  halfLifeDays: number,
  nowMs: number = Date.now(),
): number {
  if (source === "stated" || halfLifeDays <= 0) return confidence;
  const ageMs = nowMs - new Date(updatedAt).getTime();
  if (!(ageMs > 0)) return confidence;
  return round(confidence * Math.pow(0.5, ageMs / 86_400_000 / halfLifeDays));
}

export interface Inference {
  subject: string;
  predicate: Predicate;
  object: string;
  confidence: number;
  likely: boolean;
  source: "inferred" | "stated";
  because: string[];
}

/** Infer a user→concept preference from behavior (events) + felt experience
 * (loved/visited places tagged with the concept). Drill-down: if a stated
 * belief exists it pins confidence high (stated > inferred). */
export function inferConcept(
  db: DB,
  userId: string,
  concept: string,
  predicate: Predicate = "likes",
): Inference {
  const existing = db.getBelief(userId, "user", predicate, concept);
  const events = db.listEvents(userId, { concept, limit: 1000 });
  const remembered = db
    .iterRemembered(userId)
    .filter((r) => r.place.tags.includes(concept) || conceptsFromTags(r.place.tags).includes(concept));

  let signal = 0;
  const because: string[] = [];
  for (const e of events) {
    signal += EVENT_WEIGHT;
    because.push(`event#${e.id}:${e.kind}("${e.text.slice(0, 40)}")`);
  }
  for (const r of remembered) {
    signal += Math.max(0, r.aggAffect);
    because.push(`place:${r.place.name}(${r.relationship})`);
  }

  let confidence = 1 - Math.exp(-K * signal);
  let source: "inferred" | "stated" = "inferred";
  if (existing && existing.source === "stated") {
    confidence = Math.max(confidence, existing.confidence);
    source = "stated";
  }
  return {
    subject: "user",
    predicate,
    object: concept,
    confidence: round(confidence),
    likely: confidence >= LIKELY,
    source,
    because: because.slice(0, 8),
  };
}

/** Parse a natural question for a concept + polarity and answer it.
 *  "do I like coffee?" → infer likes:coffee. "do I avoid loud bars?" → avoids. */
export function ask(db: DB, userId: string, question: string): Inference & { question: string } {
  const negative = /\b(avoid|hate|dislike|don'?t like)\b/i.test(question);
  const predicate: Predicate = negative ? "avoids" : "likes";
  const concepts = extractConcepts(question);
  if (concepts.length === 0) {
    return {
      question, subject: "user", predicate, object: "", confidence: 0, likely: false,
      source: "inferred", because: ["no recognized concept in question"],
    };
  }
  return { question, ...inferConcept(db, userId, concepts[0]!, predicate) };
}

/** Promote behavior into persisted beliefs (the L0/L1 → L2 step). Scans events
 *  for concepts and intents, upserts beliefs above the consolidation threshold.
 *  Returns the beliefs written. */
export function consolidate(db: DB, userId: string): Belief[] {
  const events = db.listEvents(userId, { limit: 5000 });

  const concepts = new Set<string>();
  const intentCounts = new Map<string, number>();
  for (const e of events) {
    for (const c of e.concepts) concepts.add(c);
    for (const i of e.intents) intentCounts.set(i, (intentCounts.get(i) ?? 0) + 1);
  }
  for (const r of db.iterRemembered(userId)) for (const c of conceptsFromTags(r.place.tags)) concepts.add(c);

  const written: Belief[] = [];

  for (const concept of concepts) {
    const inf = inferConcept(db, userId, concept, "likes");
    if (inf.source === "stated") continue; // don't overwrite stated beliefs
    if (inf.confidence < CONSOLIDATE_MIN) continue;
    const belief: Belief = {
      id: null, userId, subject: "user", predicate: "likes", object: concept, otype: "concept",
      confidence: inf.confidence, support: inf.because, source: "inferred", updatedAt: nowIso(),
    };
    db.upsertBelief(belief);
    written.push(belief);
  }

  // recurring intents → a `pursues` goal belief
  for (const [purpose, count] of intentCounts) {
    if (count < 2) continue;
    const confidence = round(1 - Math.exp(-K * count * EVENT_WEIGHT));
    const belief: Belief = {
      id: null, userId, subject: "user", predicate: "pursues", object: purpose, otype: "goal",
      confidence, support: [`intent x${count}`], source: "inferred", updatedAt: nowIso(),
    };
    db.upsertBelief(belief);
    written.push(belief);
  }

  return written;
}
