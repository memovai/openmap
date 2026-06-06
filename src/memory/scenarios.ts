import { type DB, type StoredTurn } from "../store/db.js";
import { conceptsFromTags, extractConcepts } from "../nlp/extract.js";
import { ALLOWED_GOALS } from "../prompts/intent.js";
import { type ReconcileAction } from "./inference.js";
import { type Relationship, type Routine, type Scenario, nowIso } from "../core/types.js";

const ACCEPTING = new Set<Relationship>(["visited", "loved", "liked", "want_to_go"]);
const NEGATED_AFFORDANCE_RE = /^not[-_\s]?(quiet|loud|cozy|romantic|outdoor|cheap|fancy|lively)$/;
const CANONICAL_SCENARIO_CONCEPTS = new Set([
  "coffee", "tea", "ramen", "sushi", "pizza", "burger", "bar", "wine", "brunch", "dessert", "bbq",
  "vegetarian", "open_late", "walkable", "low_crowd", "crowded", "transit", "parking",
  "cozy", "quiet", "loud", "romantic", "lively", "outdoor", "cheap", "fancy",
]);
const INTENT_PRIORITY = ["work", "business", "date", "romance", "family", "study", "celebration", "hangout", "solo", "explore"];
const ROUTINE_FAMILIES = [
  { id: "business", label: "business", intents: ["business"] },
  { id: "date", label: "date", intents: ["date", "romance"] },
  { id: "family", label: "family", intents: ["family"] },
  { id: "focus", label: "focus", intents: ["work", "study"] },
  { id: "social", label: "social", intents: ["hangout", "celebration"] },
  { id: "explore", label: "explore", intents: ["solo", "explore"] },
] as const;
const ROUTINE_CONCEPT_PRIORITY = [
  "quiet", "low_crowd", "transit", "walkable", "open_late", "vegetarian", "coffee", "tea", "cozy", "outdoor",
  "romantic", "lively", "cheap", "fancy", "ramen", "sushi", "pizza", "burger", "bar", "wine", "brunch",
  "dessert", "bbq", "parking", "loud", "crowded",
];
const NEGATIVE_ROUTINE_CONCEPTS = new Set(["loud", "crowded"]);

export interface ScenarioObservation {
  userTurns: number;
  actions: Array<{ place: string; placeId: string; action: ReconcileAction; relationship: Relationship; reason: string }>;
  concepts: string[];
  intents: string[];
}

export function canonicalConcepts(items: string[]): string[] {
  const out = new Set<string>();
  for (const item of items) {
    const c = item.toLowerCase().trim();
    if (!c || NEGATED_AFFORDANCE_RE.test(c) || /^not[-_\s]?/.test(c)) continue;
    const mapped = [...conceptsFromTags([c]), ...extractConcepts(c.replace(/[_-]+/g, " "))];
    if (CANONICAL_SCENARIO_CONCEPTS.has(c)) mapped.unshift(c);
    for (const m of mapped) if (CANONICAL_SCENARIO_CONCEPTS.has(m)) out.add(m);
  }
  return [...out];
}

export function createScenarioFromObservation(userId: string, turns: StoredTurn[], observed: ScenarioObservation): Scenario | null {
  if (turns.length === 0) return null;
  const placeIds = [...new Set(observed.actions.map((a) => a.placeId))];
  const placeNames = [...new Set(observed.actions.map((a) => a.place))];
  const concepts = canonicalConcepts(observed.concepts).slice(0, 24);
  const intents = [...new Set(observed.intents)].filter((g) => ALLOWED_GOALS.includes(g)).slice(0, 16);
  const titleLead = primaryIntent(intents);
  const title = placeNames.length ? `${titleLead}: ${placeNames.slice(0, 2).join(" / ")}` : titleLead;
  const positives = observed.actions.filter((a) => ACCEPTING.has(a.relationship)).map((a) => a.place);
  const negatives = observed.actions.filter((a) => a.relationship === "disliked").map((a) => a.place);
  const actionSummary = positives.length || negatives.length
    ? [
        positives.length ? `preferred ${[...new Set(positives)].join(", ")}` : "",
        negatives.length ? `avoided ${[...new Set(negatives)].join(", ")}` : "",
      ].filter(Boolean).join("; ")
    : observed.actions.length
      ? observed.actions.map((a) => `${a.relationship} ${a.place}`).join("; ")
      : "raw conversation captured";
  const conceptSummary = concepts.length ? `; concepts ${concepts.slice(0, 8).join(", ")}` : "";
  return {
    id: null,
    userId,
    title: title.slice(0, 120),
    summary: `${observed.userTurns} user turn(s); ${actionSummary}${conceptSummary}`.slice(0, 500),
    turnIds: turns.map((t) => t.id),
    placeIds,
    concepts,
    intents,
    startedAt: turns[0]?.at ?? null,
    endedAt: turns.at(-1)?.at ?? null,
    createdAt: nowIso(),
  };
}

export function deriveRoutines(
  db: DB,
  userId: string,
  opts: { limit?: number; scenarioLimit?: number; minScenarios?: number; intent?: string; concept?: string } = {},
): Routine[] {
  const minScenarios = opts.minScenarios ?? 2;
  const scenarios = db.listScenarios(userId, { limit: opts.scenarioLimit ?? 200 });
  const groups = new Map<string, { id: string; label: string; scenarios: Scenario[] }>();
  for (const s of scenarios) {
    if (s.intents.length === 0 && s.concepts.length === 0) continue;
    const family = routineFamily(s.intents);
    const existing = groups.get(family.id) ?? { ...family, scenarios: [] };
    existing.scenarios.push(s);
    groups.set(family.id, existing);
  }

  const out: Routine[] = [];
  for (const group of groups.values()) {
    if (group.scenarios.length < minScenarios) continue;
    const conceptCounts = new Map<string, number>();
    const intentCounts = new Map<string, number>();
    const placeIds = new Set<string>();
    const positivePlaceIds = new Set<string>();
    const negativePlaceIds = new Set<string>();

    for (const s of group.scenarios) {
      for (const c of s.concepts) conceptCounts.set(c, (conceptCounts.get(c) ?? 0) + 1);
      for (const i of s.intents) intentCounts.set(i, (intentCounts.get(i) ?? 0) + 1);
      for (const placeId of s.placeIds) {
        placeIds.add(placeId);
        const rel = strongestPlaceRelationship(db, userId, placeId);
        if (rel === "disliked") negativePlaceIds.add(placeId);
        else if (rel && ACCEPTING.has(rel)) positivePlaceIds.add(placeId);
      }
    }

    const concepts = [...conceptCounts.keys()].sort(byFrequency(conceptCounts)).slice(0, 16);
    const repeatedConcepts = concepts.filter((c) => (conceptCounts.get(c) ?? 0) >= 2);
    const positiveConcepts = repeatedConcepts.filter((c) => !NEGATIVE_ROUTINE_CONCEPTS.has(c));
    const avoidConcepts = repeatedConcepts.filter((c) => NEGATIVE_ROUTINE_CONCEPTS.has(c));
    const summaryConcepts = [...positiveConcepts];
    for (const c of concepts) {
      if (summaryConcepts.length >= 4) break;
      if (NEGATIVE_ROUTINE_CONCEPTS.has(c) || summaryConcepts.includes(c)) continue;
      if (["transit", "walkable", "open_late", "vegetarian", "parking"].includes(c) || summaryConcepts.length < 2)
        summaryConcepts.push(c);
    }
    const intents = [...intentCounts.keys()].sort((a, b) => (intentCounts.get(b) ?? 0) - (intentCounts.get(a) ?? 0) || INTENT_PRIORITY.indexOf(a) - INTENT_PRIORITY.indexOf(b));
    const titleConcepts = (positiveConcepts.length ? positiveConcepts : concepts.filter((c) => !NEGATIVE_ROUTINE_CONCEPTS.has(c))).slice(0, 2);
    const title = `${group.label}: ${titleConcepts.length ? titleConcepts.map(conceptLabel).join(" + ") : "place routine"}`;
    const positiveNames = [...positivePlaceIds].map((id) => db.getPlace(id)?.name ?? id);
    const negativeNames = [...negativePlaceIds].map((id) => db.getPlace(id)?.name ?? id);
    const parts = [
      `${group.scenarios.length} related ${group.label} scenarios`,
      intents.length ? `intents ${intents.slice(0, 4).join("/")}` : "",
      summaryConcepts.length ? `looks for ${summaryConcepts.map(conceptLabel).join(", ")}` : "",
      avoidConcepts.length ? `avoids ${avoidConcepts.slice(0, 3).map(conceptLabel).join(", ")}` : "",
      positiveNames.length ? `preferred ${positiveNames.slice(0, 4).join(", ")}` : "",
      negativeNames.length ? `avoided ${negativeNames.slice(0, 4).join(", ")}` : "",
    ].filter(Boolean);
    out.push({
      id: `routine:${userId}:${group.id}`,
      userId,
      title: title.slice(0, 120),
      summary: parts.join("; ").slice(0, 700),
      scenarioIds: group.scenarios.map((s) => s.id).filter((id): id is number => id != null),
      placeIds: [...placeIds],
      positivePlaceIds: [...positivePlaceIds],
      negativePlaceIds: [...negativePlaceIds],
      concepts,
      intents,
      support: group.scenarios.length,
      confidence: round(Math.min(0.95, 0.45 + group.scenarios.length * 0.12 + repeatedConcepts.length * 0.04), 2),
      startedAt: earliestIso(group.scenarios.map((s) => s.startedAt ?? s.createdAt)),
      endedAt: latestIso(group.scenarios.map((s) => s.endedAt ?? s.createdAt)),
      updatedAt: latestIso(group.scenarios.map((s) => s.createdAt)) ?? nowIso(),
    });
  }
  return out
    .filter((r) => (opts.intent ? r.intents.includes(opts.intent) : true))
    .filter((r) => (opts.concept ? r.concepts.includes(opts.concept) : true))
    .sort((a, b) => b.confidence - a.confidence || b.support - a.support || b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, opts.limit ?? 20);
}

const round = (x: number, n = 3) => Number(x.toFixed(n));
const primaryIntent = (intents: string[]): string =>
  INTENT_PRIORITY.find((i) => intents.includes(i)) ?? intents[0] ?? "map memory";
const routineFamily = (intents: string[]): { id: string; label: string } => {
  for (const f of ROUTINE_FAMILIES) if (f.intents.some((i) => intents.includes(i))) return f;
  return { id: primaryIntent(intents), label: primaryIntent(intents) };
};
const conceptLabel = (c: string): string =>
  c === "low_crowd" ? "low crowd" : c === "open_late" ? "open late" : c === "transit" ? "near transit" : c.replace(/_/g, " ");
const byFrequency = (counts: Map<string, number>) => (a: string, b: string): number => {
  const d = (counts.get(b) ?? 0) - (counts.get(a) ?? 0);
  if (d !== 0) return d;
  const ai = ROUTINE_CONCEPT_PRIORITY.indexOf(a);
  const bi = ROUTINE_CONCEPT_PRIORITY.indexOf(b);
  return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi) || a.localeCompare(b);
};
const latestIso = (items: Array<string | null>): string | null =>
  items.filter((x): x is string => !!x).sort().at(-1) ?? null;
const earliestIso = (items: Array<string | null>): string | null =>
  items.filter((x): x is string => !!x).sort()[0] ?? null;

function strongestPlaceRelationship(db: DB, userId: string, placeId: string): Relationship | null {
  const memories = db.memoriesFor(placeId, userId);
  if (memories.length === 0) return null;
  return memories.reduce((a, b) => (Math.abs(b.affect) > Math.abs(a.affect) ? b : a)).relationship;
}
