/**
 * openmap — a map-aware memory layer for AI agents.
 *
 * The agent's conversation is the only source of memory. Places and their
 * attributes are extracted directly from what's said; there is no external POI
 * lookup. Activity (L0 events) rises into beliefs (L2 semantic graph) and a
 * persona (L3); recall and ask read that graph. See MEMORY_MODEL.md.
 */
export { OpenMap, PlaceMemory, buildOpenMap, buildMemory } from "./openmap.js";
export type {
  RememberOptions,
  MemoryExport,
  RelatedPlace,
  Anchors,
  KnowledgeGraph,
  ConversationTurn,
  ObserveResult,
  CaptureResult,
  RecallContext,
} from "./openmap.js";
export { formatPersonaContext, formatRecallBlock } from "./memory/hooks.js";

export { loadConfig, resolvedEmbedder, resolvedTagger } from "./core/config.js";
export type { Config } from "./core/config.js";

export { DB } from "./store/db.js";
export type { CollectionInfo, MemoryListItem, RememberedRow, StoredTurn } from "./store/db.js";

export { OpenAIEmbedder, getEmbedder, cosineMatrix, blendVectors } from "./nlp/embedding.js";
export type { Embedder } from "./nlp/embedding.js";

export {
  HeuristicExtractor,
  LLMExtractor,
  getExtractor,
  extractConcepts,
  conceptsFromTags,
  inferIntents,
  inferCompanion,
  inferRelationship,
  extractMeasures,
  VIBE_CONCEPTS,
} from "./nlp/extract.js";
export type { Extractor, ScoredIntent, Measure } from "./nlp/extract.js";
export { LexiconTagger, LLMTagger, getTagger, lexiconFrame } from "./nlp/tagger.js";
export type { Tagger } from "./nlp/tagger.js";
export { HeuristicMemoryExtractor, LLMMemoryExtractor, getMemoryExtractor } from "./nlp/memory-extractor.js";
export type { MemoryExtractor, ExtractedPlace } from "./nlp/memory-extractor.js";
export { OpenAILLMRunner, getRunner, extractJson } from "./nlp/llm.js";
export type { LLMRunner } from "./nlp/llm.js";

export { inferConcept, consolidate, ask, reconcileDecision, decayConfidence } from "./memory/inference.js";
export type { Inference, ReconcileAction, ReconcileDecision } from "./memory/inference.js";
export { tasteVector, effectiveTaste } from "./memory/taste.js";
export { computeAnchors, defaultAnchor } from "./memory/anchors.js";
export { frequentedAreas, primaryArea } from "./memory/regions.js";
export type { Area } from "./memory/regions.js";
export { TERMS, getCalibration, learnCalibration, allCalibrations, nearRadiusKm } from "./memory/calibration.js";
export type { Calibration, Agg, TermSpec } from "./memory/calibration.js";
export { buildGraph, graphToMermaid } from "./memory/graph.js";
export { personaPositiveText, dislikePenalty } from "./memory/persona.js";

export { derivePlaceVibe } from "./world/affordance.js";
export { relatedPlaces } from "./world/relations.js";
export { rankMemory } from "./search/ranking.js";

export { haversineKm, geoBonus, geoGate, geoAffinity } from "./core/geo.js";

export {
  RELATIONSHIPS,
  PREDICATES,
  EVENT_KINDS,
  DEFAULT_USER,
  affectFor,
  makePlaceId,
  placeTextBlob,
  rawToPlace,
  mentionToPlace,
  emptyPrefs,
  emptyFrame,
  personaPrefsSchema,
  relationshipSchema,
} from "./core/types.js";
export type {
  Relationship,
  Predicate,
  Place,
  RawPlace,
  Memory,
  OmEvent,
  Belief,
  ScoredPlace,
  GeoPoint,
  Persona,
  PersonaPrefs,
  GraphNode,
  GraphEdge,
  IntentFrame,
  IntentConstraints,
} from "./core/types.js";

export const VERSION = "0.3.0";
