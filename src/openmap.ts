import { type Config, loadConfig } from "./core/config.js";
import { DB, type CollectionInfo, type MemoryListItem, type StoredTurn, buildFtsMatch } from "./store/db.js";
import { type Embedder, getEmbedder } from "./nlp/embedding.js";
import { type Extractor, type Measure, conceptsFromTags, extractConcepts, getExtractor } from "./nlp/extract.js";
import { type MemoryExtractor, HeuristicMemoryExtractor, getMemoryExtractor } from "./nlp/memory-extractor.js";
import { type Tagger, getTagger, lexiconFrame } from "./nlp/tagger.js";
import { type LLMRunner, getRunner } from "./nlp/llm.js";
import {
  type Inference,
  type RepairAction,
  type ReconcileAction,
  ask,
  consolidate,
  decayConfidence,
  inferConcept,
  repairContradictions,
  reconcileDecision,
} from "./memory/inference.js";
import { tasteVector as tasteVectorOf } from "./memory/taste.js";
import { type Anchors, computeAnchors } from "./memory/anchors.js";
import { type Area, frequentedAreas } from "./memory/regions.js";
import { type Calibration, allCalibrations, learnCalibration } from "./memory/calibration.js";
import { buildGraph, graphToMermaid, type KnowledgeGraph } from "./memory/graph.js";
import { type RankingBeliefSignals } from "./search/ranking.js";
import { recallPlaces } from "./search/recall.js";
import { formatPersonaContext, formatRecallBlock, type RecallBlockSource } from "./memory/hooks.js";
import { type RelatedPlace, relatedPlaces as relatedPlacesOf } from "./world/relations.js";
import { ALLOWED_GOALS } from "./prompts/intent.js";
import { canonicalConcepts, createScenarioFromObservation, deriveRoutines } from "./memory/scenarios.js";
import {
  DEFAULT_USER,
  type Belief,
  type GeoPoint,
  type IntentFrame,
  type Memory,
  type OmEvent,
  type Persona,
  type PersonaPrefs,
  type Place,
  type PlaceAlias,
  type Predicate,
  type ProvenanceRef,
  type RawPlace,
  type Relationship,
  type Routine,
  type Scenario,
  type ScoredPlace,
  affectFor,
  emptyPrefs,
  mentionToPlace,
  nowIso,
  placeTextBlob,
  rawToPlace,
} from "./core/types.js";

export type { RelatedPlace } from "./world/relations.js";
export type { Anchors } from "./memory/anchors.js";
export type { KnowledgeGraph } from "./memory/graph.js";

const round = (x: number, n = 3) => Number(x.toFixed(n));
export interface RememberOptions {
  userId?: string;
  relationship?: Relationship;
  note?: string | null;
  near?: GeoPoint | null;
  companions?: string[];
  source?: string;
  sourceRefs?: ProvenanceRef[];
}
export interface MemoryExport {
  userId: string;
  memories: Memory[];
  places: Place[];
}
export interface ConversationTurn {
  role?: string;
  content: string;
  /** ISO timestamp the turn happened, if the host has it (else recording time). */
  at?: string;
}
export interface ObserveResult {
  userTurns: number;
  actions: Array<{ place: string; placeId: string; action: ReconcileAction; relationship: Relationship; reason: string }>;
  /** Calibrations auto-learned from accepted measures in the conversation. */
  learned: Measure[];
  /** Episode-level concepts extracted from the observed user turns and places. */
  concepts: string[];
  /** Episode-level intents/goals extracted from the observed user turns. */
  intents: string[];
}
/** auto-capture result: raw turns persisted to L0 + the extraction outcome. */
export interface CaptureResult {
  recorded: number; // raw turns written to the L0 log
  observed: ObserveResult | null; // null when extraction was deferred (extract:false)
  scenario: Scenario | null; // L2 episode grouping the captured turns + map memories
}
/** auto-recall result: ready-to-inject context for the host's turn loop. */
export interface RecallContext {
  /** Stable persona/geography block — cache on the system prompt. "" if unknown. */
  system: string;
  /** Per-turn relevant-places block — prepend to the user message. "" if none. */
  prepend: string;
  /** The underlying ranked places (for hosts that want structured data). */
  places: ScoredPlace[];
  /** Raw L0 turns that justify each recalled place, keyed by place id. */
  sources: Record<string, RecallSource[]>;
}
export type RecallSource = RecallBlockSource;

const ACCEPTING = new Set<Relationship>(["visited", "loved", "liked", "want_to_go"]);
const REJECTION_RE = /too far|too expensive|too pricey|too long|not worth|too much|skip it/i;

/**
 * OpenMap — a map-aware memory layer for AI agents. The agent's conversation is
 * the ONLY source of memory: places and their attributes are extracted from
 * what's said (no external POI lookup). A thin orchestrator wiring the store,
 * NLP, memory layers (taste / anchors / beliefs / graph) and the ranker; the
 * logic lives in those modules. Per-user: events (L0) → beliefs (L2) → persona.
 */
export class OpenMap {
  constructor(
    public db: DB,
    private embedder: Embedder | null,
    private extractor: Extractor,
    private tagger: Tagger,
    private halfLifeDays = 60,
    private memExtractor: MemoryExtractor = new HeuristicMemoryExtractor(),
  ) {}

  private logEvent(e: Omit<OmEvent, "id" | "createdAt">): number {
    return this.db.addEvent({ ...e, id: null, createdAt: nowIso() });
  }

  /** Build a place from a conversation mention. No POI lookup — but the name
   * itself is evidence (a place called "Blue Bottle Coffee" implies coffee), so
   * we tag it with concepts derived from its name. */
  private placeFromMention(name: string, userId = DEFAULT_USER): Place {
    const aliasTarget = this.db.resolvePlaceAlias(userId, name);
    if (aliasTarget) {
      const canonical = this.db.getPlace(aliasTarget);
      if (canonical) return canonical;
    }
    const p = rawToPlace(mentionToPlace(name));
    if (p.tags.length === 0) p.tags = extractConcepts(name);
    return p;
  }

  // ---- write path ---------------------------------------------------------
  async remember(text: string, opts: RememberOptions = {}): Promise<Memory[]> {
    const userId = opts.userId ?? DEFAULT_USER;
    const mentions = await this.extractor.extract(text);
    const textConcepts = await this.tagger.concepts(text);
    const out: Memory[] = [];
    for (const mention of mentions) {
      const place = this.placeFromMention(mention, userId);
      const mem = await this.store(place, {
        userId,
        relationship: opts.relationship ?? "mentioned",
        note: opts.note ?? text,
        companions: opts.companions ?? [],
        source: opts.source ?? "manual",
        sourceRefs: opts.sourceRefs ?? [{ kind: "stated", id: null, label: opts.source ?? "manual", snippet: text.slice(0, 220) }],
      });
      this.logEvent({
        userId, kind: "remember", text, placeId: place.id,
        concepts: [...new Set([...textConcepts, ...conceptsFromTags(place.tags)])], intents: [],
      });
      out.push(mem);
    }
    return out;
  }

  async rememberPlace(input: RawPlace | Place, opts: RememberOptions = {}): Promise<Memory> {
    const userId = opts.userId ?? DEFAULT_USER;
    const place = "id" in input ? (input as Place) : rawToPlace(input as RawPlace);
    const mem = await this.store(place, {
      userId,
      relationship: opts.relationship ?? "visited",
      note: opts.note ?? null,
      companions: opts.companions ?? [],
      source: opts.source ?? "manual",
      sourceRefs: opts.sourceRefs ?? [{ kind: "stated", id: null, label: opts.source ?? "manual" }],
    });
    this.logEvent({ userId, kind: "remember", text: place.name, placeId: place.id, concepts: conceptsFromTags(place.tags), intents: [] });
    return mem;
  }

  private async store(
    place: Place,
    o: Required<Pick<RememberOptions, "userId" | "relationship" | "note" | "companions" | "source">> & { sourceRefs?: ProvenanceRef[] },
  ): Promise<Memory> {
    const emb = this.embedder ? await this.embedder.embedOne(placeTextBlob(place)) : null;
    this.db.upsertPlace(place, emb);
    this.db.addPlaceAlias(o.userId, place.name, place.id);
    for (const alias of Array.isArray(place.raw.aliases) ? place.raw.aliases : [])
      if (typeof alias === "string" && alias.trim()) this.db.addPlaceAlias(o.userId, alias, place.id);
    const mem: Memory = {
      id: null, userId: o.userId, placeId: place.id, relationship: o.relationship,
      affect: affectFor(o.relationship), note: o.note, companions: o.companions,
      occurredAt: null, createdAt: nowIso(), source: o.source, sourceRefs: o.sourceRefs ?? [],
    };
    mem.id = this.db.addMemory(mem);
    return mem;
  }

  /**
   * Auto-capture memory from a conversation: only **user** turns are mined
   * (user-narrated facts only). Each place mention is resolved and **reconciled**
   * against existing memory — ADD / UPDATE (want_to_go → visited, sentiment flip)
   * / NOOP — instead of blindly appending. LLM mention extraction when keyed.
   */
  async observe(turns: ConversationTurn[], opts: { userId?: string; recordedTurns?: StoredTurn[] } = {}): Promise<ObserveResult> {
    const userId = opts.userId ?? DEFAULT_USER;
    const actions: ObserveResult["actions"] = [];
    const learned: Measure[] = [];
    const observedConcepts = new Set<string>();
    const observedIntents = new Set<string>();
    let userTurns = 0;
    let prev = ""; // previous turn (often the agent offering options with distances/prices)
    let recordedCursor = 0;

    for (const turn of turns) {
      const text = turn.content?.trim() ?? "";
      const recorded = text ? opts.recordedTurns?.[recordedCursor++] : undefined;
      if ((turn.role ?? "user") !== "user") {
        if (text) prev = text;
        continue;
      }
      if (!text) continue;
      userTurns++;
      const frame = await this.tagger.frame(text);
      const lexicalGoals = lexiconFrame(text).goals;
      const goals = [...new Set([...frame.goals, ...lexicalGoals])];
      for (const c of canonicalConcepts([...frame.concepts, ...frame.vibe])) observedConcepts.add(c);
      for (const g of goals) observedIntents.add(g);
      this.logEvent({
        userId, kind: "state", text, placeId: null,
        concepts: [...new Set([...frame.concepts, ...frame.vibe])], intents: goals,
      });

      const rejection = REJECTION_RE.test(`${prev} ${text}`);
      for (const ex of await this.memExtractor.extract(text, { context: prev })) {
        const place = this.placeFromMention(ex.name, userId);
        const exConcepts = canonicalConcepts(ex.concepts ?? []);
        if (exConcepts.length) place.tags = [...new Set([...place.tags, ...exConcepts])]; // richer tags from LLM
        const goalContexts = [...new Set([ex.goal, ...goals].filter((g): g is string => !!g && ALLOWED_GOALS.includes(g)))];
        if (goalContexts.length) place.tags = [...new Set([...place.tags, ...goalContexts])]; // place-side affordance learned from the user's situation
        for (const c of [...exConcepts, ...conceptsFromTags(place.tags)]) observedConcepts.add(c);
        for (const g of goalContexts) observedIntents.add(g);
        const existing = this.db.memoriesFor(place.id, userId);
        const decision = reconcileDecision(existing, ex.relationship);
        const sourceRefs: ProvenanceRef[] = recorded
          ? [{ kind: "turn", id: recorded.id, label: recorded.role, snippet: recorded.content.replace(/\s+/g, " ").slice(0, 220) }]
          : [{ kind: "stated", id: null, label: "observe", snippet: text.slice(0, 220) }];
        if (decision.action === "add") {
          await this.store(place, { userId, relationship: ex.relationship, note: text, companions: ex.companions, source: "observe", sourceRefs });
        } else if (decision.action === "update" && decision.targetId != null) {
          this.db.upsertPlace(place, this.embedder ? await this.embedder.embedOne(placeTextBlob(place)) : null);
          this.db.updateMemory(decision.targetId, { relationship: ex.relationship, affect: affectFor(ex.relationship), note: text });
        }
        if (decision.action !== "noop")
          actions.push({ place: place.name, placeId: place.id, action: decision.action, relationship: ex.relationship, reason: decision.reason });

        // revealed-preference calibration (per place, scoped to its intent context)
        if (ACCEPTING.has(ex.relationship) && !rejection) {
          for (const me of ex.measures) {
            const contexts = goalContexts.length ? goalContexts : [undefined];
            for (const context of contexts) this.learn(userId, me.term, me.value, context);
            learned.push(me);
          }
        }
        // named-area relationship extracted from text (user↔region)
        if (ex.region)
          this.db.upsertBelief({
            id: null, userId, subject: "user", predicate: "frequents", object: ex.region, otype: "region",
            confidence: 0.6,
            support: [`observed: ${place.name}`],
            provenance: [{ kind: "place", id: place.id, label: `observed ${place.name}` }],
            source: "inferred",
            updatedAt: nowIso(),
          });
      }
      prev = text;
    }
    return { userTurns, actions, learned, concepts: [...observedConcepts], intents: [...observedIntents] };
  }

  // ---- agent hooks (auto-capture / auto-recall) ---------------------------
  /**
   * auto-capture: persist the raw exchange to the L0 log (verbatim, for later
   * grounding) and — unless `extract:false` — distil it into structured memory.
   * A host can record cheaply every turn and defer the (LLM) extraction to a
   * cadence by passing `extract:false` on the cheap turns.
   */
  async capture(turns: ConversationTurn[], opts: { userId?: string; extract?: boolean } = {}): Promise<CaptureResult> {
    const userId = opts.userId ?? DEFAULT_USER;
    const recordedTurns = this.db.recordTurnsDetailed(
      userId,
      turns.map((t) => ({ role: t.role ?? "user", content: t.content ?? "", at: t.at })),
      nowIso(),
    );
    const observed = opts.extract === false ? null : await this.observe(turns, { userId, recordedTurns });
    const scenario = observed ? this.createScenario(userId, recordedTurns, observed) : null;
    return { recorded: recordedTurns.length, observed, scenario };
  }

  private createScenario(userId: string, turns: StoredTurn[], observed: ObserveResult): Scenario | null {
    const scenario = createScenarioFromObservation(userId, turns, observed);
    if (!scenario) return null;
    scenario.id = this.db.addScenario(scenario);
    return scenario;
  }

  /**
   * auto-recall: build the context a host injects before the agent answers —
   * a stable persona/geography block (for the system prompt) plus the relevant
   * remembered places for this query (to prepend to the user message).
   */
  async recallContext(
    query: string,
    opts: { near?: GeoPoint | null; limit?: number; userId?: string } = {},
  ): Promise<RecallContext> {
    const userId = opts.userId ?? DEFAULT_USER;
    const places = await this.recall(query, opts.near ?? null, opts.limit ?? 5, userId);
    const sources = Object.fromEntries(places.map((p) => [p.place.id, this.sourcesForPlace(userId, p.place)]));
    const system = formatPersonaContext(this.getPersona(userId), this.anchors(userId), this.calibrations(userId));
    return { system, prepend: formatRecallBlock(places, sources), places, sources };
  }

  private sourcesForPlace(userId: string, place: Place, limit = 2): RecallSource[] {
    const match = buildFtsMatch(place.name);
    if (!match) return [];
    const name = place.name.toLowerCase();
    const hits = this.db
      .searchTurns(userId, match, 20)
      .filter((t) => t.content.toLowerCase().includes(name))
      .sort((a, b) => Number(b.role === "user") - Number(a.role === "user"));
    const seen = new Set<number>();
    const out: RecallSource[] = [];
    for (const h of hits) {
      if (seen.has(h.id)) continue;
      seen.add(h.id);
      out.push({
        turnId: h.id,
        role: h.role,
        at: h.at,
        snippet: h.content.replace(/\s+/g, " ").slice(0, 220),
      });
      if (out.length >= limit) break;
    }
    return out;
  }

  /** Keyword (BM25) search over the raw L0 conversation log — lets the agent
   * pull the original wording to ground or verify a recalled memory. */
  searchConversation(query: string, opts: { userId?: string; limit?: number } = {}): StoredTurn[] {
    const match = buildFtsMatch(query);
    if (!match) return [];
    return this.db.searchTurns(opts.userId ?? DEFAULT_USER, match, opts.limit ?? 10);
  }

  /** Most recent raw turns from the L0 log (chronological). */
  recentConversation(opts: { userId?: string; limit?: number } = {}): StoredTurn[] {
    return this.db.recentTurns(opts.userId ?? DEFAULT_USER, opts.limit ?? 20);
  }

  /** L2 scenario summaries: one captured episode grouping raw turns, places,
   * concepts, and intents. This is the bridge from L0 logs to the L3 persona. */
  scenarios(userId = DEFAULT_USER, opts: { limit?: number; placeId?: string; intent?: string } = {}): Scenario[] {
    return this.db.listScenarios(userId, opts);
  }

  /** Long-horizon routines derived from repeated L2 scenarios. These are not
   * stored separately: they are replayable rollups from scenario history. */
  routines(
    userId = DEFAULT_USER,
    opts: { limit?: number; scenarioLimit?: number; minScenarios?: number; intent?: string; concept?: string } = {},
  ): Routine[] {
    return deriveRoutines(this.db, userId, opts);
  }

  private strongestPlaceRelationship(userId: string, placeId: string): Relationship | null {
    const memories = this.db.memoriesFor(placeId, userId);
    if (memories.length === 0) return null;
    return memories.reduce((a, b) => (Math.abs(b.affect) > Math.abs(a.affect) ? b : a)).relationship;
  }

  // ---- taste --------------------------------------------------------------
  tasteVector(userId = DEFAULT_USER): Float32Array | null {
    return tasteVectorOf(this.db, userId);
  }

  // ---- read path ----------------------------------------------------------
  /**
   * Search the user's remembered places by *resolved intent* via **hybrid
   * retrieval**: a keyword arm (FTS5/BM25, always on) + a vector arm (cosine,
   * only when an embedder is configured) fused with Reciprocal Rank Fusion, then
   * blended with the personal signals (affect, taste, vibe, proximity, dislikes).
   * Without an embedder it gracefully degrades to keyword-only. The query is also
   * logged as a behavioral event (it's conversation too), feeding inference.
   */
  async recall(query: string, near: GeoPoint | null = null, limit = 5, userId = DEFAULT_USER): Promise<ScoredPlace[]> {
    const prefs = this.db.getPersonaPrefs(userId).prefs ?? emptyPrefs();
    const beliefSignals = this.rankingBeliefSignals(userId);
    const { frame, places } = await recallPlaces({
      db: this.db,
      embedder: this.embedder,
      tagger: this.tagger,
      userId,
      query,
      near,
      limit,
      prefs,
      beliefSignals,
    });
    this.logEvent({
      userId, kind: "search", text: query, placeId: null,
      concepts: [...new Set([...frame.concepts, ...frame.vibe])], intents: frame.goals,
    });
    return places;
  }

  private rankingBeliefSignals(userId: string): RankingBeliefSignals {
    const terms = (predicate: Predicate) =>
      this.beliefs(userId, { predicate, minConfidence: 0.3 })
        .filter((b) => b.otype === "concept" || b.otype === "goal")
        .map((b) => ({ term: b.object, confidence: b.confidence }));
    return { likes: terms("likes"), avoids: terms("avoids"), pursues: terms("pursues") };
  }

  /** Teach the user's personal meaning of a fuzzy term from one accepted sample
   * (revealed preference): e.g. learn("u","near",3) — they accepted a place 3km
   * away, so "near" ≥ 3km. Terms: near|walk_time|budget|noise|crowd|transit_walk. */
  learn(userId: string, term: string, sample: number, context?: string): void {
    learnCalibration(this.db, userId, term, sample, context);
  }

  /** The user's learned calibrations (what their fuzzy words mean). */
  calibrations(userId = DEFAULT_USER): Calibration[] {
    return allCalibrations(this.db, userId);
  }

  /** Convenience alias for `learn(userId, "near", km)`. */
  learnNearRadius(userId: string, km: number): void {
    learnCalibration(this.db, userId, "near", km);
  }

  /** Intents inferred from a query (proactive purpose guessing). */
  async intents(query: string): Promise<string[]> {
    return (await this.tagger.frame(query)).goals;
  }

  /** Resolve a maps query into its latent situational intent frame. */
  async resolveIntent(query: string): Promise<IntentFrame> {
    return this.tagger.frame(query);
  }

  // ---- inference (the "do I like coffee?" loop) ---------------------------
  ask(question: string, userId = DEFAULT_USER): Inference & { question: string } {
    return ask(this.db, userId, question);
  }
  infer(concept: string, predicate: Predicate = "likes", userId = DEFAULT_USER): Inference {
    return inferConcept(this.db, userId, concept, predicate);
  }
  /** Promote behavior into persisted beliefs (L0/L1 → L2). */
  consolidate(userId = DEFAULT_USER): Belief[] {
    return consolidate(this.db, userId);
  }
  /** Repair inferred contradictions in old DBs, e.g. both likes and avoids loud. */
  repairContradictions(userId = DEFAULT_USER): RepairAction[] {
    return repairContradictions(this.db, userId);
  }
  /** Beliefs with recency decay applied (inferred beliefs fade; stated don't). */
  beliefs(userId = DEFAULT_USER, opts: { predicate?: Predicate; minConfidence?: number } = {}): Belief[] {
    const min = opts.minConfidence ?? 0;
    return this.db
      .listBeliefs(userId, { predicate: opts.predicate })
      .map((b) => ({ ...b, confidence: decayConfidence(b.confidence, b.updatedAt, b.source, this.halfLifeDays) }))
      .filter((b) => b.confidence >= min)
      .sort((a, b) => b.confidence - a.confidence);
  }

  // ---- relations & anchors ------------------------------------------------
  /** Set a place-role relation (home/work) as a stated belief. */
  setPlaceRole(userId: string, placeId: string, role: "home" | "work"): Belief {
    const belief: Belief = {
      id: null, userId, subject: "user", predicate: role === "home" ? "lives_near" : "works_near",
      object: placeId, otype: "place", confidence: 0.99,
      support: ["stated"],
      provenance: [{ kind: "stated", id: placeId, label: role }],
      source: "stated",
      updatedAt: nowIso(),
    };
    this.db.upsertBelief(belief);
    return belief;
  }

  anchors(userId = DEFAULT_USER): Anchors {
    return computeAnchors(this.db, userId);
  }

  /** Areas the user is active in (user↔area relationship), derived from where
   * their remembered places cluster. Picking places near X strengthens X. */
  regions(userId = DEFAULT_USER, opts: { cellKm?: number; limit?: number } = {}): Area[] {
    return frequentedAreas(this.db, userId, opts);
  }

  relatedPlaces(placeId: string, opts: { limit?: number; radiusKm?: number } = {}): RelatedPlace[] {
    return relatedPlacesOf(this.db, placeId, opts);
  }

  addPlaceAlias(userId: string, alias: string, placeId: string): PlaceAlias {
    return this.db.addPlaceAlias(userId, alias, placeId);
  }

  placeAliases(userId = DEFAULT_USER, placeId?: string): PlaceAlias[] {
    return placeId ? this.db.aliasesForPlace(userId, placeId) : this.db.listPlaceAliases(userId);
  }

  // ---- personal knowledge graph -------------------------------------------
  graph(userId = DEFAULT_USER): KnowledgeGraph {
    return buildGraph(this.db, userId);
  }
  graphMermaid(userId = DEFAULT_USER): string {
    return graphToMermaid(buildGraph(this.db, userId));
  }

  // ---- persona ------------------------------------------------------------
  getPersona(userId = DEFAULT_USER): Persona {
    const { prefs, updatedAt } = this.db.getPersonaPrefs(userId);
    const objs = (p: Predicate, min: number) => this.beliefs(userId, { predicate: p, minConfidence: min }).map((b) => b.object);
    return {
      userId, stated: prefs ?? emptyPrefs(),
      derived: { likes: objs("likes", 0.4), avoids: objs("avoids", 0.4), pursues: objs("pursues", 0.3) },
      updatedAt,
    };
  }

  setPersona(userId = DEFAULT_USER, patch: Partial<PersonaPrefs>): Persona {
    const current = this.db.getPersonaPrefs(userId).prefs ?? emptyPrefs();
    const merged: PersonaPrefs = { ...current, ...patch };
    this.db.setPersonaPrefs(userId, merged);
    // mirror explicit prefs into stated beliefs so the graph stays unified
    for (const l of merged.likes)
      this.db.upsertBelief({ id: null, userId, subject: "user", predicate: "likes", object: l, otype: "concept", confidence: 0.95, support: ["stated"], provenance: [{ kind: "stated", id: null, label: `likes ${l}` }], source: "stated", updatedAt: nowIso() });
    for (const d of merged.dislikes)
      this.db.upsertBelief({ id: null, userId, subject: "user", predicate: "avoids", object: d, otype: "concept", confidence: 0.95, support: ["stated"], provenance: [{ kind: "stated", id: null, label: `avoids ${d}` }], source: "stated", updatedAt: nowIso() });
    return this.getPersona(userId);
  }

  clearPersona(userId = DEFAULT_USER): void {
    this.db.deletePersona(userId);
  }

  tasteProfile(userId = DEFAULT_USER): Record<string, unknown> {
    const rows = this.db.iterRemembered(userId);
    const favorites = rows
      .filter((r) => r.aggAffect > 0)
      .sort((a, b) => b.aggAffect - a.aggAffect)
      .slice(0, 10)
      .map((r) => ({ name: r.place.name, affect: round(r.aggAffect, 2), relationship: r.relationship, tags: r.place.tags }));
    return {
      userId, persona: this.getPersona(userId),
      topBeliefs: this.beliefs(userId, { minConfidence: 0.4 }).slice(0, 10).map((b) => ({ predicate: b.predicate, object: b.object, confidence: b.confidence, source: b.source })),
      hasTasteVector: this.tasteVector(userId) !== null,
      favorites,
      ...this.db.stats(userId),
    };
  }

  // ---- management ---------------------------------------------------------
  listEvents(userId = DEFAULT_USER, opts: { kind?: string; concept?: string; limit?: number } = {}): OmEvent[] {
    return this.db.listEvents(userId, opts);
  }
  listMemories(userId = DEFAULT_USER, opts: { relationship?: Relationship; limit?: number } = {}): MemoryListItem[] {
    return this.db.listMemories(userId, opts);
  }
  updateMemory(id: number, fields: { relationship?: Relationship; affect?: number; note?: string | null }): boolean {
    return this.db.updateMemory(id, fields);
  }
  forget(userId: string, target: { memoryId?: number; placeId?: string }): number {
    if (target.memoryId != null) return this.db.deleteMemory(target.memoryId, userId);
    if (target.placeId != null) return this.db.forgetPlace(userId, target.placeId);
    return 0;
  }
  listPlaces(userId = DEFAULT_USER, opts: { tag?: string; limit?: number } = {}): Place[] {
    return this.db.listPlaces(userId, opts);
  }
  exportMemories(userId = DEFAULT_USER): MemoryExport {
    const items = this.db.listMemories(userId, { limit: 1_000_000 });
    const places = new Map<string, Place>();
    for (const it of items) if (it.place) places.set(it.place.id, it.place);
    return { userId, memories: items.map((i) => i.memory), places: [...places.values()] };
  }
  async importMemories(data: MemoryExport, userId = data.userId): Promise<number> {
    const byId = new Map(data.places.map((p) => [p.id, p]));
    let n = 0;
    for (const mem of data.memories) {
      const place = byId.get(mem.placeId);
      if (!place) continue;
      this.db.upsertPlace(place, this.embedder ? await this.embedder.embedOne(placeTextBlob(place)) : null);
      this.db.addMemory({ ...mem, id: null, userId });
      n++;
    }
    return n;
  }

  // ---- collections --------------------------------------------------------
  collectionList(userId = DEFAULT_USER): CollectionInfo[] {
    return this.db.collectionList(userId);
  }
  collectionAdd(userId: string, name: string, placeId: string): void {
    this.db.collectionAddItem(this.db.collectionUpsert(userId, name), placeId);
  }
  collectionRemove(userId: string, name: string, placeId: string): number {
    const col = this.db.collectionByName(userId, name);
    return col ? this.db.collectionRemoveItem(col.id, placeId) : 0;
  }
  collectionShow(userId: string, name: string): Place[] {
    const col = this.db.collectionByName(userId, name);
    return col ? this.db.collectionItems(col.id) : [];
  }
}

/** Backwards-compatible alias. */
export const PlaceMemory = OpenMap;

/** Build an OpenMap from config. Optionally inject an `llm` runner so extraction
 * borrows the host agent's model instead of openmap's own (BYOC). */
export function buildOpenMap(cfg: Config = loadConfig(), opts: { llm?: LLMRunner } = {}): OpenMap {
  const runner = opts.llm ?? getRunner(cfg);
  return new OpenMap(
    new DB(cfg.dbPath),
    getEmbedder(cfg),
    getExtractor(cfg, runner),
    getTagger(cfg, runner),
    cfg.beliefHalfLifeDays,
    getMemoryExtractor(cfg, runner),
  );
}
export const buildMemory = buildOpenMap;
