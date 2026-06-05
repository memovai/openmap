import { test } from "node:test";
import assert from "node:assert/strict";

import { DB } from "../src/store/db.js";
import { type Embedder } from "../src/nlp/embedding.js";
import { HeuristicExtractor, type ScoredIntent } from "../src/nlp/extract.js";
import { type Tagger, lexiconFrame } from "../src/nlp/tagger.js";
import { OpenMap, buildOpenMap } from "../src/openmap.js";
import { loadConfig } from "../src/core/config.js";
import { type LLMRunner } from "../src/nlp/llm.js";
import { LLMTagger } from "../src/nlp/tagger.js";
import { decayConfidence, reconcileDecision } from "../src/memory/inference.js";
import { getCalibration } from "../src/memory/calibration.js";
import { type ExtractedPlace, type MemoryExtractor } from "../src/nlp/memory-extractor.js";
import { type Memory, type RawPlace } from "../src/core/types.js";

/** Deterministic stand-in for the LLM tagger (tests stay offline). */
class FakeTagger implements Tagger {
  async concepts(text: string): Promise<string[]> {
    const s = text.toLowerCase();
    const out: string[] = [];
    if (/coffee|espresso|latte|cafe|flat white/.test(s)) out.push("coffee");
    if (/ramen|noodle/.test(s)) out.push("ramen");
    if (/wine/.test(s)) out.push("wine");
    if (/\bbar\b|sports|pub/.test(s)) out.push("bar");
    return out;
  }
  async intents(text: string): Promise<ScoredIntent[]> {
    const s = text.toLowerCase();
    return /romantic|candlelit|\bdate\b/.test(s)
      ? [{ purpose: "date", score: 2 }, { purpose: "romance", score: 1 }]
      : [];
  }
  async frame(text: string) {
    const s = text.toLowerCase();
    const vibe = ["cozy", "quiet", "romantic", "lively", "outdoor"].filter((v) => s.includes(v));
    const constraints: { dietary?: string[] } = {};
    if (/vegetarian|vegan/.test(s)) constraints.dietary = ["vegetarian"];
    return {
      rawQuery: text,
      goals: (await this.intents(text)).map((i) => i.purpose),
      companions: /partner|girlfriend|boyfriend|\bdate\b/.test(s) ? "partner" : /parents|mom|dad|family/.test(s) ? "parents" : null,
      occasion: null,
      concepts: await this.concepts(text),
      vibe,
      constraints,
    };
  }
}

/** Deterministic stand-in for a real embedder (the product has no offline
 * embedder; tests inject this so the vector arm + taste are exercised offline). */
class FakeEmbedder implements Embedder {
  dim = 64;
  private vec(text: string): Float32Array {
    const v = new Float32Array(this.dim);
    const toks = (text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
    for (const t of toks) {
      let h = 2166136261;
      for (let i = 0; i < t.length; i++) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
      v[(h >>> 0) % this.dim]! += 1;
    }
    let n = 0;
    for (let i = 0; i < this.dim; i++) n += v[i]! * v[i]!;
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < this.dim; i++) v[i]! /= n;
    return v;
  }
  async embed(texts: string[]) { return texts.map((t) => this.vec(t)); }
  async embedOne(text: string) { return this.vec(text); }
}

// Conversation is the only source — no provider. The agent hands structured
// places (with tags/coords) via rememberPlace, or free text via remember/observe.
function makeMemory(): OpenMap {
  return new OpenMap(new DB(":memory:"), new FakeEmbedder(), new HeuristicExtractor(), new FakeTagger());
}
const raw = (name: string, lat: number, lng: number, tags: string[]): RawPlace => ({
  name, lat, lng, category: null, address: null, source: "agent", sourceId: name, tags, raw: {},
});

// ---- write + recall --------------------------------------------------------
test("remember (from text) and recall — no POI lookup", async () => {
  const mem = makeMemory();
  const out = await mem.remember('I loved "Blue Bottle Coffee"', { relationship: "loved" });
  assert.equal(out.length, 1);
  assert.equal(out[0]!.affect, 1.0);
  const res = await mem.recall("coffee");
  assert.equal(res[0]!.place.name, "Blue Bottle Coffee");
  assert.equal(res[0]!.place.source, "mention"); // not fetched from any POI source
});

test("recall ranks remembered places by affect + taste", async () => {
  const mem = makeMemory();
  await mem.rememberPlace(raw("Blue Bottle Coffee", 35.6, 139.7, ["cafe", "coffee_shop"]), { relationship: "mentioned" });
  await mem.rememberPlace(raw("Quiet Garden Bistro", 35.61, 139.71, ["cozy", "outdoor", "wine"]), { relationship: "loved" });
  await mem.rememberPlace(raw("Loud Sports Bar", 35.62, 139.72, ["sports", "loud", "beer"]), { relationship: "mentioned" });
  const names = (await mem.recall("a place to eat", null, 5)).map((r) => r.place.name);
  assert.ok(names.indexOf("Quiet Garden Bistro") < names.indexOf("Loud Sports Bar"));
});

test("recall boosts places whose vibe affordances match the resolved intent", async () => {
  const mem = makeMemory();
  await mem.rememberPlace(raw("Quiet Garden Bistro", 35.61, 139.71, ["cozy", "outdoor", "wine"]), { relationship: "mentioned" });
  await mem.rememberPlace(raw("Loud Sports Bar", 35.62, 139.72, ["sports", "loud", "beer"]), { relationship: "mentioned" });
  const res = await mem.recall("a cozy outdoor spot", null, 5);
  assert.ok((res.find((r) => r.place.name === "Quiet Garden Bistro")!.reasons.vibeBonus as number) > 1);
  assert.equal(res.find((r) => r.place.name === "Loud Sports Bar")!.reasons.vibeBonus, 1);
});

test("recall on empty memory is safe", async () => {
  assert.deepEqual(await makeMemory().recall("anything"), []);
});

// ---- behavioral inference (searched coffee → likely likes coffee) ----------
test("infers coffee preference from search behavior (queries are conversation)", async () => {
  const mem = makeMemory();
  await mem.recall("coffee shops", null, 5, "a");
  await mem.recall("best espresso near me", null, 5, "a");
  const ans = mem.ask("do I like coffee?", "a");
  assert.equal(ans.object, "coffee");
  assert.equal(ans.likely, true);
  assert.ok(ans.because.length >= 1);
});

test("one search is 'possible' not 'likely'; a loved place is strong", async () => {
  const mem = makeMemory();
  await mem.recall("coffee", null, 5, "a");
  assert.equal(mem.ask("do I like coffee?", "a").likely, false);
  await mem.remember('loved "Blue Bottle Coffee"', { userId: "b", relationship: "loved" });
  const strong = mem.ask("do I like coffee?", "b");
  assert.ok(strong.confidence >= 0.5);
  assert.equal(strong.likely, true);
});

// ---- intent resolution -----------------------------------------------------
test("intent inference: romantic → date/romance", async () => {
  const purposes = await makeMemory().intents("a romantic candlelit dinner");
  assert.ok(purposes.includes("date"));
  assert.ok(purposes.includes("romance"));
});

test("resolveIntent fills the latent frame (indirect maps intent)", async () => {
  const f = await makeMemory().resolveIntent("a romantic dinner with my partner");
  assert.ok(f.goals.includes("date"));
  assert.equal(f.companions, "partner");
  assert.ok(f.vibe.includes("romantic"));
});

test("lexicon frame handles negation + companion/goal contradiction", () => {
  const f = lexiconFrame("romantic dinner with my parents, nothing too fancy");
  assert.equal(f.companions, "parents");
  assert.ok(!f.goals.includes("date"));
  assert.ok(!f.vibe.includes("fancy"));
  assert.notEqual(f.constraints.maxBudget, "high");
});

test("recall logs the query as a behavioral event (concepts + intents)", async () => {
  const mem = makeMemory();
  await mem.recall("romantic coffee shop", null, 5, "a");
  const ev = mem.listEvents("a", { kind: "search" })[0]!;
  assert.ok(ev.intents.includes("date"));
  assert.ok(ev.concepts.includes("coffee"));
});

// ---- reconcile + observe ---------------------------------------------------
test("reconcile decides add / noop / want→visited update / sentiment flip", () => {
  const m = (relationship: string, id = 1, affect = 0): Memory => ({
    id, userId: "a", placeId: "p", relationship: relationship as Memory["relationship"],
    affect, note: null, companions: [], occurredAt: null, createdAt: "", source: "",
  });
  assert.equal(reconcileDecision([], "visited").action, "add");
  assert.equal(reconcileDecision([m("visited")], "visited").action, "noop");
  const planned = reconcileDecision([m("want_to_go", 7, 0.4)], "visited");
  assert.equal(planned.action, "update");
  assert.equal(planned.targetId, 7);
  const flip = reconcileDecision([m("loved", 3, 1)], "disliked");
  assert.equal(flip.action, "update");
  assert.equal(flip.targetId, 3);
});

test("observe auto-captures from conversation and reconciles (no duplicates)", async () => {
  const mem = makeMemory();
  await mem.observe(
    [
      { role: "user", content: 'I really want to go to "Ramen Nagi"' },
      { role: "assistant", content: "great choice!" },
    ],
    { userId: "u" },
  );
  let mems = mem.listMemories("u");
  assert.equal(mems.length, 1);
  assert.equal(mems[0]!.memory.relationship, "want_to_go");

  const res = await mem.observe([{ role: "user", content: 'went to "Ramen Nagi" last night and loved it' }], { userId: "u" });
  mems = mem.listMemories("u");
  assert.equal(mems.length, 1);
  assert.equal(mems[0]!.memory.relationship, "loved");
  assert.ok(res.actions.some((a) => a.action === "update"));
});

test("auto-capture logs raw turns to L0 and conversation_search recalls them", async () => {
  const mem = makeMemory();
  const res = await mem.capture(
    [
      { role: "user", content: 'loved "Quiet Beans" — perfect for getting work done' },
      { role: "assistant", content: "noted!" },
    ],
    { userId: "u" },
  );
  assert.equal(res.recorded, 2); // both raw turns logged
  assert.equal(res.observed?.actions.length, 1); // and extracted
  // the original wording is retrievable for grounding
  const hits = mem.searchConversation("work done", { userId: "u" });
  assert.ok(hits.some((t) => /Quiet Beans/.test(t.content)));
  // raw log is per-user
  assert.equal(mem.searchConversation("work done", { userId: "other" }).length, 0);
});

test("auto-capture with extract:false only logs raw turns (defers extraction)", async () => {
  const mem = makeMemory();
  const res = await mem.capture([{ role: "user", content: 'thinking about "Brew Lab"' }], { userId: "u", extract: false });
  assert.equal(res.recorded, 1);
  assert.equal(res.observed, null);
  assert.equal(mem.listMemories("u").length, 0); // nothing extracted yet
  assert.equal(mem.recentConversation({ userId: "u" }).length, 1); // but logged
});

test("auto-recall builds an injectable persona block + relevant-places block", async () => {
  const mem = makeMemory();
  mem.setPersona("u", { likes: ["coffee"], dislikes: ["loud bars"] });
  await mem.observe([{ role: "user", content: 'loved "Quiet Beans" for focused work' }], { userId: "u" });
  const ctx = await mem.recallContext("somewhere to get work done", { userId: "u", limit: 3 });
  assert.match(ctx.system, /user-place-profile/);
  assert.match(ctx.system, /coffee/); // stated like surfaced
  assert.match(ctx.system, /loud bars/); // stated dislike surfaced
  assert.match(ctx.prepend, /recalled-places/);
  assert.match(ctx.prepend, /Quiet Beans/);
  assert.ok(ctx.places.length >= 1);
});

test("auto-recall persona/prepend blocks are empty when nothing is known", async () => {
  const mem = makeMemory();
  const ctx = await mem.recallContext("dinner", { userId: "fresh" });
  assert.equal(ctx.system, ""); // no persona/geo → no noise injected
  assert.equal(ctx.prepend, ""); // no remembered places
});

// ---- consolidation + knowledge graph --------------------------------------
test("consolidate promotes behavior into beliefs", async () => {
  const mem = makeMemory();
  await mem.recall("coffee", null, 5, "c");
  await mem.recall("espresso bar", null, 5, "c");
  const written = mem.consolidate("c");
  assert.ok(written.some((b) => b.object === "coffee" && b.predicate === "likes"));
  assert.ok(mem.beliefs("c", { predicate: "likes" }).some((b) => b.object === "coffee"));
});

test("personal knowledge graph + Mermaid expose user→concept edges", async () => {
  const mem = makeMemory();
  await mem.recall("coffee", null, 5, "c");
  await mem.recall("flat white", null, 5, "c");
  mem.consolidate("c");
  const g = mem.graph("c");
  assert.ok(g.nodes.some((n) => n.type === "concept" && n.label === "coffee"));
  assert.ok(g.edges.some((e) => e.predicate === "likes" && e.target === "concept:coffee"));
  const mmd = mem.graphMermaid("c");
  assert.match(mmd, /graph LR/);
  assert.match(mmd, /coffee/);
});

// ---- relations, anchors, persona ------------------------------------------
test("place roles + default anchor for unanchored queries", async () => {
  const mem = makeMemory();
  const m = await mem.rememberPlace(raw("Ramen Nagi", 35.66, 139.7, ["ramen"]), { userId: "u", relationship: "visited" });
  mem.setPlaceRole("u", m.placeId, "home");
  assert.equal(mem.anchors("u").home?.name, "Ramen Nagi");
  const res = await mem.recall("noodles", null, 5, "u");
  assert.ok(res[0]!.distanceKm != null); // geo-gated around learned home
});

test("observe auto-learns calibration from accepted measures (revealed preference)", async () => {
  const mem = makeMemory();
  // agent offered a place 3km away; user accepts → learn near = 3 (no explicit call)
  const res = await mem.observe(
    [
      { role: "assistant", content: '"Ramen Nagi" is about 3km away' },
      { role: "user", content: 'went to "Ramen Nagi" and loved it' },
    ],
    { userId: "u" },
  );
  assert.equal(mem.anchors("u").nearRadiusKm, 3);
  assert.ok(res.learned.some((m) => m.term === "near" && m.value === 3));
});

test("mention extraction ignores contraction apostrophes", async () => {
  const mem = makeMemory();
  const out = await mem.remember('let\'s do "Ritual Coffee", loved it', { relationship: "loved" });
  assert.equal(out.length, 1);
  assert.equal(mem.listMemories()[0]!.place!.name, "Ritual Coffee"); // not "s do"
});

test("observe learns context-conditioned calibration from the turn's intent", async () => {
  const mem = makeMemory();
  await mem.observe(
    [
      { role: "assistant", content: '"Bella" is 5km away' },
      { role: "user", content: 'romantic dinner at "Bella", loved it' },
    ],
    { userId: "u" },
  );
  assert.equal(getCalibration(mem.db, "u", "near", "date").value, 5); // near-for-a-date learned
});

test("structured extraction: per-place sentiment + named region (LLM-shaped)", async () => {
  // a stand-in for the LLM memory extractor: one message, two places, opposite sentiment
  const fake: MemoryExtractor = {
    async extract(): Promise<ExtractedPlace[]> {
      return [
        { name: "Bella", relationship: "loved", companions: [], region: "Shibuya", measures: [{ term: "near", value: 4 }], goal: "date" },
        { name: "Noisy Joe", relationship: "disliked", companions: [], region: null, measures: [], goal: "date" },
      ];
    },
  };
  const mem = new OpenMap(new DB(":memory:"), new FakeEmbedder(), new HeuristicExtractor(), new FakeTagger(), 60, fake);
  await mem.observe([{ role: "user", content: "date night: Bella in Shibuya was lovely, Noisy Joe was awful" }], { userId: "u" });
  const rels = Object.fromEntries(mem.listMemories("u").map((it) => [it.place!.name, it.memory.relationship]));
  assert.equal(rels["Bella"], "loved");
  assert.equal(rels["Noisy Joe"], "disliked"); // per-place, not one sentence-level label
  assert.ok(mem.beliefs("u", { predicate: "frequents" }).some((b) => b.object === "Shibuya")); // user↔region
  assert.equal(getCalibration(mem.db, "u", "near", "date").value, 4); // accepted 4km for a date
});

test("observe does not learn from a rejection", async () => {
  const mem = makeMemory();
  await mem.observe(
    [
      { role: "assistant", content: '"Ramen Nagi" is 8km away' },
      { role: "user", content: "8km is too far, skip it" },
    ],
    { userId: "v" },
  );
  assert.equal(mem.anchors("v").nearRadiusKm, 2); // unchanged prior — rejection
});

test("LLM extraction runs through an injectable runner (host model / BYOC)", async () => {
  // a fake runner stands in for the host agent's model (or a BYOC endpoint)
  const runner: LLMRunner = {
    async run({ json }) {
      return json
        ? JSON.stringify({ goals: ["date"], companions: "partner", concepts: ["coffee"], vibe: ["romantic"], constraints: {} })
        : "";
    },
  };
  // component-level: the tagger uses the runner, not its own client
  const f = await new LLMTagger(runner, "any-model").frame("anything");
  assert.ok(f.goals.includes("date"));
  assert.equal(f.companions, "partner");
  // end-to-end: inject the runner into buildOpenMap (no API key needed)
  const om = buildOpenMap({ ...loadConfig({}), dbPath: ":memory:" }, { llm: runner });
  assert.ok((await om.resolveIntent("anything")).goals.includes("date"));
});

test("calibration layer learns personal meaning of fuzzy terms", () => {
  const mem = makeMemory();
  const cal = (u: string, t: string) => mem.calibrations(u).find((c) => c.term === t)!;
  // "near": tolerance (max accepted) — the flagship case
  assert.equal(mem.anchors("u").nearRadiusKm, 2); // prior before learning
  mem.learn("u", "near", 3); // accepted a place 3km away → "near" ≥ 3km
  assert.equal(mem.anchors("u").nearRadiusKm, 3);
  mem.learn("u", "near", 1); // a closer accept doesn't shrink the tolerance
  assert.equal(cal("u", "near").value, 3);
  // generic terms: walk_time (max), budget (ema → first sample sets it)
  mem.learn("u", "walk_time", 12);
  assert.equal(cal("u", "walk_time").value, 12);
  mem.learn("u", "budget", 500);
  assert.equal(cal("u", "budget").value, 500);
  // unknown terms are ignored (no hardcoded behavior)
  mem.learn("u", "bogus", 9);
  assert.equal(mem.calibrations("u").find((c) => c.term === "bogus"), undefined);
});

test("calibration is context-conditioned (near-for-date ≠ near-for-coffee)", () => {
  const mem = makeMemory();
  mem.learn("u", "near", 5, "date"); // willing to travel 5km for a date
  mem.learn("u", "near", 1, "coffee"); // but only 1km for coffee
  assert.equal(getCalibration(mem.db, "u", "near", "date").value, 5);
  assert.equal(getCalibration(mem.db, "u", "near", "coffee").value, 1);
  // unknown context falls back to the global value (max tolerance across contexts)
  assert.equal(getCalibration(mem.db, "u", "near", "work").value, 5);
  // context-scoped calibrations are surfaced
  assert.ok(mem.calibrations("u").some((c) => c.term === "near@date" && c.value === 5));
});

test("frequented areas capture the user↔area relationship + drive default anchor", async () => {
  const mem = makeMemory();
  // two places clustered downtown + one far → the busy cluster is the user's area
  await mem.rememberPlace(raw("Cafe A", 35.600, 139.700, ["cafe"]), { userId: "u", relationship: "loved" });
  await mem.rememberPlace(raw("Cafe B", 35.601, 139.701, ["cafe"]), { userId: "u", relationship: "visited" });
  await mem.rememberPlace(raw("Far Diner", 35.900, 139.900, ["restaurant"]), { userId: "u", relationship: "visited" });
  const areas = mem.regions("u");
  assert.ok(areas.length >= 2);
  assert.equal(areas[0]!.count, 2); // busiest area first
  assert.ok(Math.abs(areas[0]!.lat - 35.6005) < 0.01);
  // an unanchored query defaults to the most-active area (distance computed)
  const res = await mem.recall("coffee", null, 5, "u");
  assert.ok(res[0]!.distanceKm != null);
});

test("related places: near via vector + geo", async () => {
  const mem = makeMemory();
  await mem.rememberPlace(raw("Blue Bottle Coffee", 35.6, 139.7, ["cafe", "coffee_shop"]), { relationship: "loved" });
  await mem.rememberPlace(raw("Ramen Nagi", 35.66, 139.7, ["ramen"]), { relationship: "visited" });
  const blue = mem.listPlaces().find((p) => p.name === "Blue Bottle Coffee")!;
  assert.ok(mem.relatedPlaces(blue.id, { radiusKm: 10 }).some((r) => r.relations.includes("near")));
});

test("persona steers recall; dislikes penalize", async () => {
  const mem = makeMemory();
  mem.setPersona("u1", { likes: ["cozy", "wine", "outdoor"], dislikes: ["sports", "loud"] });
  await mem.rememberPlace(raw("Quiet Garden Bistro", 35.61, 139.71, ["cozy", "outdoor", "wine"]), { userId: "u1", relationship: "mentioned" });
  await mem.rememberPlace(raw("Loud Sports Bar", 35.62, 139.72, ["sports", "loud", "beer"]), { userId: "u1", relationship: "mentioned" });
  const res = await mem.recall("a place to eat", null, 5, "u1");
  const names = res.map((r) => r.place.name);
  assert.ok(names.indexOf("Quiet Garden Bistro") < names.indexOf("Loud Sports Bar"));
  assert.ok((res.find((r) => r.place.name === "Loud Sports Bar")!.reasons.dislikePenalty as number) < 1);
});

test("stated persona becomes a belief in the graph", () => {
  const mem = makeMemory();
  mem.setPersona("u1", { likes: ["coffee"] });
  assert.ok(mem.beliefs("u1", { predicate: "likes" }).some((b) => b.object === "coffee" && b.source === "stated"));
});

test("persona merge keeps prior fields", () => {
  const mem = makeMemory();
  mem.setPersona("a", { likes: ["wine"] });
  const p = mem.setPersona("a", { dislikes: ["loud"] });
  assert.deepEqual(p.stated.likes, ["wine"]);
  assert.deepEqual(p.stated.dislikes, ["loud"]);
});

// ---- isolation + management + storage -------------------------------------
test("everything is isolated per user", async () => {
  const mem = makeMemory();
  await mem.remember('"Ramen Nagi"', { userId: "a", relationship: "loved" });
  assert.equal((await mem.recall("ramen", null, 5, "a")).length, 1);
  assert.equal((await mem.recall("ramen", null, 5, "b")).length, 0);
  assert.equal(mem.listMemories("b").length, 0);
});

test("forget, collections, export/import", async () => {
  const mem = makeMemory();
  const out = await mem.remember('"Ramen Nagi"', { userId: "a", relationship: "loved" });
  mem.collectionAdd("a", "tokyo", out[0]!.placeId);
  assert.equal(mem.collectionShow("a", "tokyo")[0]!.name, "Ramen Nagi");
  const dump = mem.exportMemories("a");
  assert.equal(await mem.importMemories(dump, "b"), 1);
  assert.equal(mem.listMemories("b").length, 1);
  assert.equal(mem.forget("a", { memoryId: out[0]!.id! }), 1);
  assert.equal(mem.listMemories("a").length, 0);
});

test("sqlite-vec store is enabled and powers KNN", async () => {
  const mem = makeMemory();
  assert.equal(mem.db.vecEnabled, true);
  await mem.rememberPlace(raw("Blue Bottle Coffee", 35.6, 139.7, ["cafe"]), { relationship: "loved" });
  const blue = mem.listPlaces()[0]!;
  const knn = mem.db.searchPlaceVectors(mem.db.embeddingFor(blue.id)!, 5);
  assert.equal(knn[0]!.placeId, blue.id);
  assert.ok(knn[0]!.score > 0.99);
});

test("inferred beliefs decay with recency; stated do not", () => {
  const old = new Date(Date.now() - 60 * 86_400_000).toISOString();
  assert.ok(Math.abs(decayConfidence(0.8, old, "inferred", 60) - 0.4) < 0.02);
  assert.equal(decayConfidence(0.8, old, "stated", 60), 0.8);
  assert.equal(decayConfidence(0.8, old, "inferred", 0), 0.8);
});
