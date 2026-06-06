import { type DB, type RememberedRow } from "../store/db.js";
import { type Embedder, cosineMatrix } from "../nlp/embedding.js";
import { type Tagger } from "../nlp/tagger.js";
import { type Anchors, computeAnchors } from "../memory/anchors.js";
import { getCalibration, nearRadiusKm } from "../memory/calibration.js";
import { effectiveTaste } from "../memory/taste.js";
import {
  type GeoPoint,
  type IntentFrame,
  type Place,
  type RawPlace,
  type Relationship,
  type ScoredPlace,
  affectFor,
  emptyPrefs,
  placeTextBlob,
  rawToPlace,
} from "../core/types.js";
import { rankMemory, type RankingBeliefSignals } from "./ranking.js";

export interface CandidatePlaceInput {
  name: string;
  lat?: number | null;
  lng?: number | null;
  category?: string | null;
  address?: string | null;
  source?: string;
  sourceId?: string | null;
  tags?: string[];
  raw?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PlaceSearchPlan {
  query: string;
  frame: IntentFrame;
  searchQuery: string;
  include: string[];
  avoid: string[];
  location: GeoPoint | null;
  anchor: "explicit" | "home" | "work" | "usual_area" | null;
  nearRadiusKm: number;
  constraints: IntentFrame["constraints"] & {
    noiseMax?: number | null;
    crowdMax?: number | null;
    transitWalkMax?: number | null;
    walkTimeMax?: number | null;
  };
  profile: {
    likes: string[];
    avoids: string[];
    pursues: string[];
  };
  rememberedPlaces: Array<{
    name: string;
    relationship: Relationship;
    tags: string[];
  }>;
  rationale: string[];
}

export interface CandidateMemoryMatch {
  matched: boolean;
  placeId: string | null;
  relationship: Relationship | null;
  affect: number;
}

export interface RankedCandidatePlace extends ScoredPlace {
  inputIndex: number;
  memory: CandidateMemoryMatch;
}

export interface CandidateRankingResult {
  query: string;
  plan: PlaceSearchPlan;
  results: RankedCandidatePlace[];
}

export interface SearchAssistArgs {
  db: DB;
  tagger: Tagger;
  userId: string;
  query: string;
  near?: GeoPoint | null;
}

export interface CandidateRankingArgs extends SearchAssistArgs {
  embedder: Embedder | null;
  candidates: CandidatePlaceInput[];
  limit: number;
  beliefSignals?: RankingBeliefSignals;
  plan?: PlaceSearchPlan;
}

const round = (x: number, n = 3) => Number(x.toFixed(n));
const VIBE_TERMS = new Set(["cozy", "quiet", "loud", "romantic", "lively", "outdoor", "fancy", "cheap"]);
const LOW_CROWD_TERMS = new Set(["low_crowd", "uncrowded", "calm"]);
const LOUD_TERMS = new Set(["loud", "noisy", "noise"]);
const CROWDED_TERMS = new Set(["crowded", "busy", "packed"]);
const NEGATIVE_AFFORDANCE_TERMS = new Set(["loud", "noisy", "noise", "crowded", "busy", "packed", "touristy", "cramped", "expensive"]);

export async function buildPlaceSearchPlan(args: SearchAssistArgs): Promise<PlaceSearchPlan> {
  const { db, tagger, userId, query, near = null } = args;
  const baseFrame = await tagger.frame(query);
  const profile = memoryProfile(db, userId);
  const frame = completeFrameWithMemory(baseFrame, profile);
  const context = frame.goals[0];
  const anchors = computeAnchors(db, userId);
  const { location, anchor } = resolveLocation(anchors, near);
  const constraints: PlaceSearchPlan["constraints"] = {
    ...frame.constraints,
    noiseMax: getCalibration(db, userId, "noise", context).value,
    crowdMax: getCalibration(db, userId, "crowd", context).value,
    transitWalkMax: getCalibration(db, userId, "transit_walk", context).value,
    walkTimeMax: getCalibration(db, userId, "walk_time", context).value,
  };
  const include = unique([
    ...frame.concepts,
    ...frame.vibe,
    ...frame.goals,
    ...constraintTerms(frame),
    ...profile.likes.slice(0, 8),
    ...profile.pursues.slice(0, 4),
  ]).filter((t) => !profile.avoids.includes(t));
  const avoid = unique(profile.avoids.slice(0, 10));
  const extra = include.filter((t) => !query.toLowerCase().includes(t.replace(/_/g, " "))).slice(0, 6);
  const searchQuery = uniqueWords([query, ...extra]).join(" ");
  const rememberedPlaces = db
    .iterRemembered(userId)
    .filter((r) => Math.abs(r.aggAffect) > 0)
    .sort((a, b) => Math.abs(b.aggAffect) - Math.abs(a.aggAffect))
    .slice(0, 6)
    .map((r) => ({ name: r.place.name, relationship: r.relationship, tags: r.place.tags.slice(0, 6) }));
  const rationale = [
    include.length ? `bias search toward ${include.slice(0, 6).join(", ")}` : "",
    avoid.length ? `avoid ${avoid.slice(0, 6).join(", ")}` : "",
    location ? `rank around ${anchor ?? "explicit location"} within user's near≈${nearRadiusKm(db, userId, context)}km` : "",
  ].filter(Boolean);
  return {
    query,
    frame,
    searchQuery,
    include,
    avoid,
    location,
    anchor,
    nearRadiusKm: nearRadiusKm(db, userId, context),
    constraints,
    profile,
    rememberedPlaces,
    rationale,
  };
}

export async function rankCandidatePlaces(args: CandidateRankingArgs): Promise<CandidateRankingResult> {
  const { db, embedder, userId, query, near = null, candidates, limit, beliefSignals } = args;
  const plan = args.plan ?? await buildPlaceSearchPlan(args);
  const normalized = candidates.map((candidate, inputIndex) => enrichCandidateWithMemory(db, userId, normalizeCandidate(candidate), inputIndex));
  if (normalized.length === 0) return { query, plan, results: [] };

  const queryText = [plan.searchQuery, ...plan.include, ...constraintTerms(plan.frame)].join(" ");
  let candidateEmbeddings: Array<Float32Array | null> = normalized.map(() => null);
  let vectorRelevance: number[] = normalized.map(() => 0);
  let tasteSim: number[] = normalized.map(() => 0);

  if (embedder) {
    const texts = normalized.map((x) => placeTextBlob(x.place));
    const embedded = await embedder.embed(texts);
    candidateEmbeddings = embedded;
    vectorRelevance = cosineMatrix(await embedder.embedOne(queryText), embedded);
    const taste = await effectiveTaste(db, embedder, userId, db.getPersonaPrefs(userId).prefs ?? emptyPrefs());
    if (taste) tasteSim = cosineMatrix(taste, embedded);
  }

  const lexical = normalized.map((x) => lexicalRelevance(queryText, x.place));
  const relevance = new Map<string, number>();
  for (let i = 0; i < normalized.length; i++) {
    const id = normalized[i]!.place.id;
    relevance.set(id, Math.max(0, lexical[i] ?? 0) + Math.max(0, vectorRelevance[i] ?? 0));
  }

  const rows: RememberedRow[] = normalized.map((x, i) => ({
    place: x.place,
    embedding: candidateEmbeddings[i] ?? null,
    aggAffect: x.memory.affect,
    relationship: x.memory.relationship ?? "mentioned",
  }));
  const ranked = rankMemory({
    items: rows,
    relevance,
    tasteSim,
    prefs: db.getPersonaPrefs(userId).prefs ?? emptyPrefs(),
    beliefSignals,
    frame: plan.frame,
    near: near ?? plan.location,
    nearRadiusKm: plan.nearRadiusKm,
    constraintLimits: {
      noiseMax: plan.constraints.noiseMax,
      crowdMax: plan.constraints.crowdMax,
      transitWalkMax: plan.constraints.transitWalkMax,
      walkTimeMax: plan.constraints.walkTimeMax,
    },
    limit,
  });
  const byId = new Map(normalized.map((x) => [x.place.id, x]));
  return {
    query,
    plan,
    results: ranked.map((r) => {
      const source = byId.get(r.place.id)!;
      return {
        ...r,
        relationship: source.memory.relationship,
        inputIndex: source.inputIndex,
        memory: source.memory,
        reasons: {
          ...r.reasons,
          memoryMatch: source.memory.matched,
          rememberedPlaceId: source.memory.placeId ?? "none",
          lexicalRelevance: round(lexical[source.inputIndex] ?? 0),
        },
      };
    }),
  };
}

function memoryProfile(db: DB, userId: string): PlaceSearchPlan["profile"] {
  const prefs = db.getPersonaPrefs(userId).prefs ?? emptyPrefs();
  const beliefs = (predicate: "likes" | "avoids" | "pursues") =>
    db
      .listBeliefs(userId, { predicate })
      .filter((b) => b.confidence >= 0.3 && (b.otype === "concept" || b.otype === "goal"))
      .map((b) => b.object);
  const positiveTags = tagCounts(db.iterRemembered(userId).filter((r) => r.aggAffect > 0.2));
  const negativeTags = tagCounts(db.iterRemembered(userId).filter((r) => r.aggAffect < -0.2)).filter((t) => NEGATIVE_AFFORDANCE_TERMS.has(t));
  return {
    likes: unique([...prefs.likes, ...prefs.vibes, ...prefs.dietary, ...beliefs("likes"), ...positiveTags]),
    avoids: unique([...prefs.dislikes, ...beliefs("avoids"), ...negativeTags]),
    pursues: unique([...beliefs("pursues")]),
  };
}

function tagCounts(rows: RememberedRow[]): string[] {
  const counts = new Map<string, number>();
  for (const row of rows) {
    for (const tag of row.place.tags) counts.set(tag, (counts.get(tag) ?? 0) + Math.max(0.1, Math.abs(row.aggAffect)));
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([tag]) => tag)
    .slice(0, 8);
}

function completeFrameWithMemory(frame: IntentFrame, profile: PlaceSearchPlan["profile"]): IntentFrame {
  const out: IntentFrame = {
    ...frame,
    goals: [...frame.goals],
    concepts: [...frame.concepts],
    vibe: [...frame.vibe],
    constraints: { ...frame.constraints, dietary: frame.constraints.dietary ? [...frame.constraints.dietary] : undefined },
  };
  for (const term of profile.likes) {
    if (VIBE_TERMS.has(term)) out.vibe = unique([...out.vibe, term]);
    else if (!LOW_CROWD_TERMS.has(term)) out.concepts = unique([...out.concepts, term]);
  }
  if (!out.constraints.noise && (profile.avoids.some((t) => LOUD_TERMS.has(t)) || profile.likes.includes("quiet"))) out.constraints.noise = "quiet";
  if (!out.constraints.crowd && (profile.avoids.some((t) => CROWDED_TERMS.has(t)) || profile.likes.some((t) => LOW_CROWD_TERMS.has(t)))) out.constraints.crowd = "low";
  if (!out.constraints.dietary?.length) {
    const dietary = profile.likes.filter((t) => t === "vegetarian" || t === "vegan");
    if (dietary.length) out.constraints.dietary = dietary;
  }
  return out;
}

function resolveLocation(anchors: Anchors, near: GeoPoint | null): { location: GeoPoint | null; anchor: PlaceSearchPlan["anchor"] } {
  if (near) return { location: near, anchor: "explicit" };
  if (anchors.home?.lat != null && anchors.home.lng != null) return { location: { lat: anchors.home.lat, lng: anchors.home.lng }, anchor: "home" };
  if (anchors.work?.lat != null && anchors.work.lng != null) return { location: { lat: anchors.work.lat, lng: anchors.work.lng }, anchor: "work" };
  if (anchors.usualArea) return { location: { lat: anchors.usualArea.lat, lng: anchors.usualArea.lng }, anchor: "usual_area" };
  return { location: null, anchor: null };
}

function normalizeCandidate(input: CandidatePlaceInput): Place {
  const raw = { ...(input.raw ?? {}) };
  for (const [k, v] of Object.entries(input)) {
    if (!["name", "lat", "lng", "category", "address", "source", "sourceId", "tags", "raw"].includes(k)) raw[k] = v;
  }
  const r: RawPlace = {
    name: input.name,
    lat: typeof input.lat === "number" ? input.lat : null,
    lng: typeof input.lng === "number" ? input.lng : null,
    category: typeof input.category === "string" ? input.category : null,
    address: typeof input.address === "string" ? input.address : null,
    source: typeof input.source === "string" ? input.source : "candidate",
    sourceId: typeof input.sourceId === "string" ? input.sourceId : null,
    tags: Array.isArray(input.tags) ? input.tags.map(String) : [],
    raw,
  };
  return rawToPlace(r);
}

function enrichCandidateWithMemory(db: DB, userId: string, candidate: Place, inputIndex: number): { place: Place; inputIndex: number; memory: CandidateMemoryMatch } {
  const knownId = db.resolvePlaceAlias(userId, candidate.name) ?? (db.getPlace(candidate.id) ? candidate.id : null);
  const memories = knownId ? db.memoriesFor(knownId, userId) : [];
  const known = knownId ? db.getPlace(knownId) : null;
  const affect = memories.length ? memories.reduce((s, m) => s + m.affect, 0) / memories.length : 0;
  const relationship = memories.length ? memories.reduce((a, b) => (Math.abs(b.affect) > Math.abs(a.affect) ? b : a)).relationship : null;
  const place = known
    ? {
        ...candidate,
        tags: unique([...candidate.tags, ...known.tags]),
        raw: { ...known.raw, ...candidate.raw },
      }
    : candidate;
  return {
    place,
    inputIndex,
    memory: { matched: memories.length > 0, placeId: knownId, relationship, affect },
  };
}

function lexicalRelevance(queryText: string, place: Place): number {
  const query = tokens(queryText);
  if (!query.length) return 0;
  const target = new Set(tokens(placeTextBlob(place)));
  const hits = query.filter((t) => target.has(t)).length;
  return hits / query.length;
}

function constraintTerms(frame: IntentFrame): string[] {
  const c = frame.constraints;
  return [
    ...(c.openNow ? ["open_late"] : []),
    ...(c.walkable ? ["walkable"] : []),
    ...(c.dietary ?? []),
    ...(c.maxBudget ? [c.maxBudget === "low" ? "cheap" : c.maxBudget === "high" ? "fancy" : "mid_budget"] : []),
    ...(c.noise === "quiet" ? ["quiet"] : c.noise === "loud" ? ["loud"] : []),
    ...(c.crowd === "low" ? ["low_crowd", "uncrowded"] : c.crowd === "high" ? ["crowded"] : []),
    ...(c.travelMode === "walk" ? ["walkable"] : c.travelMode === "transit" ? ["transit", "station"] : c.travelMode === "drive" ? ["parking"] : []),
  ];
}

function tokens(text: string): string[] {
  return unique(text.toLowerCase().replace(/_/g, " ").match(/[a-z0-9]+/g) ?? []).filter((t) => t.length > 1);
}

function unique<T>(xs: T[]): T[] {
  return [...new Set(xs.filter((x) => x != null && String(x).trim()))];
}

function uniqueWords(xs: string[]): string[] {
  return unique(xs.join(" ").replace(/_/g, " ").match(/[A-Za-z0-9'-]+/g) ?? []);
}
