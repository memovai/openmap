import { createHash } from "node:crypto";
import { z } from "zod";

export const DEFAULT_USER = "default";

export const nowIso = (): string => new Date().toISOString();

// ---- relationships (user→place edges) -------------------------------------
export const RELATIONSHIPS = [
  "loved",
  "liked",
  "visited",
  "want_to_go",
  "disliked",
  "mentioned",
] as const;
export type Relationship = (typeof RELATIONSHIPS)[number];
export const relationshipSchema = z.enum(RELATIONSHIPS);

export const AFFECT_BY_RELATIONSHIP: Record<Relationship, number> = {
  loved: 1.0,
  liked: 0.5,
  want_to_go: 0.4,
  visited: 0.2,
  mentioned: 0.0,
  disliked: -1.0,
};
export const affectFor = (rel: Relationship): number => AFFECT_BY_RELATIONSHIP[rel] ?? 0;

// ---- world model: places (shared, objective) ------------------------------
export interface Place {
  id: string;
  name: string;
  lat: number | null;
  lng: number | null;
  category: string | null;
  address: string | null;
  source: string;
  sourceId: string | null;
  tags: string[];
  raw: Record<string, unknown>;
}

export function makePlaceId(
  source: string,
  sourceId: string | null,
  name: string,
  lat: number | null,
  lng: number | null,
): string {
  if (sourceId) return `${source}:${sourceId}`;
  const seed = `${name.trim().toLowerCase()}|${(lat ?? 0).toFixed(5)}|${(lng ?? 0).toFixed(5)}`;
  return "h:" + createHash("sha1").update(seed).digest("hex").slice(0, 16);
}

export function placeTextBlob(p: Place): string {
  return [p.name, p.category ?? "", p.address ?? "", p.tags.join(" ")]
    .filter((s) => s && s.trim())
    .join(" — ")
    .trim();
}

/** A place as extracted from conversation (or handed in by the host agent),
 * before it becomes a canonical Place. Conversation is the only source — there
 * is no external POI lookup; coords/address are filled only if stated. */
export interface RawPlace {
  name: string;
  lat: number | null;
  lng: number | null;
  category: string | null;
  address: string | null;
  source: string;
  sourceId: string | null;
  tags: string[];
  raw: Record<string, unknown>;
}

export function rawToPlace(r: RawPlace): Place {
  return {
    id: makePlaceId(r.source, r.sourceId, r.name, r.lat, r.lng),
    name: r.name, lat: r.lat, lng: r.lng, category: r.category, address: r.address,
    source: r.source, sourceId: r.sourceId, tags: r.tags, raw: r.raw,
  };
}

/** Turn a bare mention from conversation into a named place (no POI lookup). */
export function mentionToPlace(name: string): RawPlace {
  return { name, lat: null, lng: null, category: null, address: null, source: "mention", sourceId: null, tags: [], raw: {} };
}

// ---- L0: episodic events --------------------------------------------------
export const EVENT_KINDS = ["search", "discover", "visit", "ask", "remember", "state"] as const;
export type EventKind = (typeof EVENT_KINDS)[number];

export interface OmEvent {
  id: number | null;
  userId: string;
  kind: EventKind;
  text: string;
  placeId: string | null;
  concepts: string[];
  intents: string[];
  createdAt: string;
}

// ---- L2: beliefs (personal knowledge graph edges) -------------------------
export const PREDICATES = [
  "likes",
  "avoids",
  "prefers",
  "visited",
  "lives_near",
  "works_near",
  "pursues",
  "frequents",
] as const;
export type Predicate = (typeof PREDICATES)[number];

export type ObjectType = "concept" | "place" | "goal" | "person" | "region";
export type BeliefSource = "inferred" | "stated";

export interface Belief {
  id: number | null;
  userId: string;
  subject: string; // usually "user"
  predicate: Predicate;
  object: string; // concept string | place id | goal | person
  otype: ObjectType;
  confidence: number; // 0..1
  support: string[]; // provenance: event/memory refs
  source: BeliefSource;
  updatedAt: string;
}

// ---- L0/L1 edge: memories (explicit user→place relationships) -------------
export interface Memory {
  id: number | null;
  userId: string;
  placeId: string;
  relationship: Relationship;
  affect: number;
  note: string | null;
  companions: string[];
  occurredAt: string | null;
  createdAt: string;
  source: string;
}

// ---- L3: persona ----------------------------------------------------------
export interface PersonaPrefs {
  likes: string[];
  dislikes: string[];
  vibes: string[];
  dietary: string[];
  budget: "low" | "mid" | "high" | null;
  notes: string | null;
}

export const emptyPrefs = (): PersonaPrefs => ({
  likes: [],
  dislikes: [],
  vibes: [],
  dietary: [],
  budget: null,
  notes: null,
});

export const personaPrefsSchema = z
  .object({
    likes: z.array(z.string()),
    dislikes: z.array(z.string()),
    vibes: z.array(z.string()),
    dietary: z.array(z.string()),
    budget: z.enum(["low", "mid", "high"]).nullable(),
    notes: z.string().nullable(),
  })
  .partial();

export interface Persona {
  userId: string;
  stated: PersonaPrefs; // user-set
  derived: { likes: string[]; avoids: string[]; pursues: string[] }; // distilled from L2
  updatedAt: string | null;
}

// ---- intent (working memory): the latent situational frame behind a query --
export interface IntentConstraints {
  openNow?: boolean;
  maxBudget?: "low" | "mid" | "high" | null;
  dietary?: string[];
  walkable?: boolean;
}

/** A maps query is rarely literal ("romantic coffee shop" → a date). The frame
 * is the structured, memory-completed intent the ranker actually searches. */
export interface IntentFrame {
  rawQuery: string;
  goals: string[]; // date, work, family, celebration, business, hangout, solo, explore
  companions: string | null; // alone | partner | kids | friends | client | parents
  occasion: string | null; // tonight, weekend, anniversary, …
  concepts: string[]; // coffee, ramen, … (category-ish)
  vibe: string[]; // cozy, quiet, lively, not-touristy, instagrammable …
  constraints: IntentConstraints;
}

export const emptyFrame = (rawQuery: string): IntentFrame => ({
  rawQuery,
  goals: [],
  companions: null,
  occasion: null,
  concepts: [],
  vibe: [],
  constraints: {},
});

// ---- search results -------------------------------------------------------
export interface ScoredPlace {
  place: Place;
  score: number;
  distanceKm: number | null;
  relationship: Relationship | null;
  reasons: Record<string, number | boolean | string>;
}

export interface GeoPoint {
  lat: number;
  lng: number;
}

// ---- graph view -----------------------------------------------------------
export interface GraphNode {
  id: string;
  type: "user" | ObjectType;
  label: string;
}
export interface GraphEdge {
  source: string;
  predicate: string;
  target: string;
  confidence?: number;
}
