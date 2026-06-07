import { type DB, buildFtsMatch } from "../store/db.js";
import { type Embedder, cosineMatrix } from "../nlp/embedding.js";
import { type Tagger } from "../nlp/tagger.js";
import { type GeoPoint, type IntentFrame, type PersonaPrefs, type ScoredPlace, emptyPrefs } from "../core/types.js";
import { constraintConceptTerms } from "../core/vocabulary.js";
import { defaultAnchor } from "../memory/anchors.js";
import { getCalibration, nearRadiusKm } from "../memory/calibration.js";
import { effectiveTaste } from "../memory/taste.js";
import { rankMemory, rrfMerge, type RankingBeliefSignals } from "./ranking.js";

export interface RecallPipelineArgs {
  db: DB;
  embedder: Embedder | null;
  tagger: Tagger;
  userId: string;
  query: string;
  near: GeoPoint | null;
  limit: number;
  prefs?: PersonaPrefs;
  beliefSignals?: RankingBeliefSignals;
}

export interface RecallPipelineResult {
  frame: IntentFrame;
  places: ScoredPlace[];
}

export async function recallPlaces(args: RecallPipelineArgs): Promise<RecallPipelineResult> {
  const { db, embedder, tagger, userId, query, near, limit, beliefSignals } = args;
  const frame = await tagger.frame(query);
  const items = db.iterRemembered(userId);
  if (items.length === 0) return { frame, places: [] };
  const remembered = new Set(items.map((i) => i.place.id));
  const queryText = [frame.rawQuery, ...frame.vibe, ...frame.concepts, ...frame.goals, ...constraintQueryTerms(frame)].join(" ");

  const match = buildFtsMatch(queryText);
  const kw = match ? db.searchPlaceFts(match, 50).filter((id) => remembered.has(id)) : [];

  let vec: string[] = [];
  let tasteSim: number[] = items.map(() => 0);
  const prefs = args.prefs ?? db.getPersonaPrefs(userId).prefs ?? emptyPrefs();
  const taste = await effectiveTaste(db, embedder, userId, prefs);
  if (embedder && items.some((i) => i.embedding)) {
    const qv = await embedder.embedOne(queryText);
    const sims = cosineMatrix(qv, items.map((i) => i.embedding));
    vec = items
      .map((it, i) => ({ id: it.place.id, s: it.embedding ? sims[i]! : -1 }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .map((x) => x.id);
    if (taste) tasteSim = cosineMatrix(taste, items.map((i) => i.embedding));
  }

  const relevance = rrfMerge([kw, vec]);
  const anchor = near ?? defaultAnchor(db, userId);
  const context = frame.goals[0];
  return {
    frame,
    places: rankMemory({
      items,
      relevance,
      tasteSim,
      prefs,
      beliefSignals,
      frame,
      near: anchor,
      nearRadiusKm: nearRadiusKm(db, userId, context),
      constraintLimits: {
        noiseMax: getCalibration(db, userId, "noise", context).value,
        crowdMax: getCalibration(db, userId, "crowd", context).value,
        transitWalkMax: getCalibration(db, userId, "transit_walk", context).value,
        walkTimeMax: getCalibration(db, userId, "walk_time", context).value,
      },
      limit,
    }),
  };
}

function constraintQueryTerms(frame: IntentFrame): string[] {
  return constraintConceptTerms(frame.constraints);
}
