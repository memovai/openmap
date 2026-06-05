// MCP server exposing openmap as agent tools. Run: `openmap serve-mcp`.
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { z } from "zod";
import { buildOpenMap } from "./openmap.js";
import { type Place, type ScoredPlace, personaPrefsSchema, relationshipSchema } from "./core/types.js";

const placeBrief = (p: Place) => ({
  name: p.name, placeId: p.id, category: p.category, address: p.address, lat: p.lat, lng: p.lng, tags: p.tags,
});
const scored = (items: ScoredPlace[]) =>
  items.map((s) => ({ ...placeBrief(s.place), score: s.score, distanceKm: s.distanceKm, relationship: s.relationship, reasons: s.reasons }));
const near = (lat?: number, lng?: number) => (lat != null && lng != null ? { lat, lng } : null);
const asText = (obj: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(obj, null, 2) }] });

export async function main(): Promise<void> {
  let McpServer: any, StdioServerTransport: any;
  try {
    ({ McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js"));
    ({ StdioServerTransport } = await import("@modelcontextprotocol/sdk/server/stdio.js"));
  } catch {
    console.error("MCP support needs the optional dependency:\n  npm install @modelcontextprotocol/sdk");
    process.exit(1);
  }

  const mem = buildOpenMap();
  const server = new McpServer({ name: "openmap", version: "0.3.0" });

  server.tool(
    "remember",
    "Store place memories extracted from free text (e.g. 'loved Blue Bottle in Tokyo'). Also logs an episodic event.",
    { text: z.string(), relationship: relationshipSchema.optional(), note: z.string().optional(), lat: z.number().optional(), lng: z.number().optional(), userId: z.string().optional() },
    async (a: any) => {
      const out = await mem.remember(a.text, { userId: a.userId, relationship: a.relationship ?? "mentioned", note: a.note ?? null, near: near(a.lat, a.lng) });
      return asText({ stored: out.length, places: out.map((m) => m.placeId) });
    },
  );

  server.tool(
    "recall",
    "Search the user's remembered places by resolved intent (taste + vibe + proximity). Returns the inferred intent frame plus ranked places.",
    { query: z.string(), lat: z.number().optional(), lng: z.number().optional(), limit: z.number().optional(), userId: z.string().optional() },
    async (a: any) => {
      const frame = await mem.resolveIntent(a.query);
      const results = await mem.recall(a.query, near(a.lat, a.lng), a.limit ?? 5, a.userId);
      return asText({ frame, results: scored(results) });
    },
  );

  server.tool(
    "resolve_intent",
    "Resolve a maps query into its latent situational intent frame (goals, companions, vibe, constraints) — maps intent is rarely literal.",
    { query: z.string() },
    async (a: any) => asText(await mem.resolveIntent(a.query)),
  );

  server.tool(
    "observe",
    "Auto-capture memory from a conversation: mine the USER's turns for place mentions and reconcile them (add/update) into memory. Pass recent messages.",
    {
      messages: z.array(z.object({ role: z.string().optional(), content: z.string() })),
      userId: z.string().optional(),
    },
    async (a: any) => asText(await mem.observe(a.messages, { userId: a.userId })),
  );

  server.tool(
    "capture",
    "auto-capture: log the raw turns to the L0 conversation history (for later grounding) AND extract/reconcile place memory. Use after each exchange. Set extract:false to only log raw turns (defer extraction).",
    {
      messages: z.array(z.object({ role: z.string().optional(), content: z.string() })),
      extract: z.boolean().optional(),
      userId: z.string().optional(),
    },
    async (a: any) => asText(await mem.capture(a.messages, { userId: a.userId, extract: a.extract })),
  );

  server.tool(
    "recall_context",
    "auto-recall: build the context to inject before answering — a stable persona/geography block (system) + the relevant remembered places for this query (prepend to the user message). One call replaces recall+persona+anchors for the turn loop.",
    { query: z.string(), lat: z.number().optional(), lng: z.number().optional(), limit: z.number().optional(), userId: z.string().optional() },
    async (a: any) => {
      const ctx = await mem.recallContext(a.query, { near: near(a.lat, a.lng), limit: a.limit ?? 5, userId: a.userId });
      return asText({ system: ctx.system, prepend: ctx.prepend, places: scored(ctx.places) });
    },
  );

  server.tool(
    "conversation_search",
    "Search the raw conversation history (L0, BM25 keyword) to recall the user's original wording — use to ground or verify a recalled memory ('when did they say they loved X').",
    { query: z.string(), limit: z.number().optional(), userId: z.string().optional() },
    async (a: any) => asText({ turns: mem.searchConversation(a.query, { userId: a.userId, limit: a.limit ?? 10 }) }),
  );

  server.tool(
    "ask",
    "Answer a preference question by inferring from behavior, e.g. 'do I like coffee?'. Drills down to events + loved places.",
    { question: z.string(), userId: z.string().optional() },
    async (a: any) => asText(mem.ask(a.question, a.userId)),
  );

  server.tool(
    "consolidate",
    "Promote the user's behavior (events) into persisted beliefs. Run periodically.",
    { userId: z.string().optional() },
    async (a: any) => asText({ written: mem.consolidate(a.userId) }),
  );

  server.tool(
    "beliefs",
    "List the user's semantic beliefs (personal knowledge-graph edges: likes/avoids/lives_near/pursues...).",
    { predicate: z.string().optional(), minConfidence: z.number().optional(), userId: z.string().optional() },
    async (a: any) => asText({ beliefs: mem.beliefs(a.userId, { predicate: a.predicate, minConfidence: a.minConfidence }) }),
  );

  server.tool(
    "graph",
    "Return the user's personal knowledge graph as nodes + edges plus a Mermaid diagram.",
    { userId: z.string().optional() },
    async (a: any) => asText({ ...mem.graph(a.userId), mermaid: mem.graphMermaid(a.userId) }),
  );

  server.tool(
    "anchors",
    "The user's learned spatial self-model (home/work/usual area + their 'near' radius). Use to default 'near me' when a query gives no location.",
    { userId: z.string().optional() },
    async (a: any) => asText(mem.anchors(a.userId)),
  );

  server.tool(
    "calibrate",
    "Teach what a fuzzy place term means to this user from one accepted sample (revealed preference). term: near (km) | walk_time (min) | budget (spend) | noise (0..1). E.g. they picked a place 3km away → calibrate near 3.",
    {
      term: z.enum(["near", "walk_time", "budget", "noise"]),
      value: z.number(),
      context: z.string().optional(),
      userId: z.string().optional(),
    },
    async (a: any) => {
      mem.learn(a.userId ?? "default", a.term, a.value, a.context);
      return asText({ calibrations: mem.calibrations(a.userId) });
    },
  );

  server.tool(
    "calibrations",
    "The user's learned meaning of fuzzy terms (near/walk_time/budget/noise) — their personal place vocabulary.",
    { userId: z.string().optional() },
    async (a: any) => asText({ calibrations: mem.calibrations(a.userId) }),
  );

  server.tool(
    "regions",
    "Areas the user is active in (the user↔area relationship), clustered from their place activity — useful for resolving 'near me' / 'my area'.",
    { userId: z.string().optional() },
    async (a: any) => asText({ regions: mem.regions(a.userId) }),
  );

  server.tool(
    "taste_profile",
    "Return the user's persona, top beliefs, favorites and stats.",
    { userId: z.string().optional() },
    async (a: any) => asText(mem.tasteProfile(a.userId)),
  );

  server.tool(
    "get_persona",
    "Get the user's stated + derived preferences.",
    { userId: z.string().optional() },
    async (a: any) => asText(mem.getPersona(a.userId)),
  );

  server.tool(
    "set_persona",
    "Merge explicit preferences into the user's persona (also written as stated beliefs).",
    { likes: z.array(z.string()).optional(), dislikes: z.array(z.string()).optional(), vibes: z.array(z.string()).optional(), dietary: z.array(z.string()).optional(), budget: z.enum(["low", "mid", "high"]).nullable().optional(), notes: z.string().nullable().optional(), userId: z.string().optional() },
    async (a: any) => { const { userId, ...patch } = a; return asText(mem.setPersona(userId, personaPrefsSchema.parse(patch))); },
  );

  server.tool(
    "set_place_role",
    "Record a place role for the user as a stated belief (home → lives_near, work → works_near).",
    { placeId: z.string(), role: z.enum(["home", "work"]), userId: z.string().optional() },
    async (a: any) => asText(mem.setPlaceRole(a.userId ?? "default", a.placeId, a.role)),
  );

  server.tool(
    "list_memories",
    "List the user's stored memories.",
    { relationship: relationshipSchema.optional(), limit: z.number().optional(), userId: z.string().optional() },
    async (a: any) => {
      const items = mem.listMemories(a.userId, { relationship: a.relationship, limit: a.limit ?? 50 });
      return asText({ count: items.length, memories: items.map((it) => ({ id: it.memory.id, relationship: it.memory.relationship, affect: it.memory.affect, note: it.memory.note, place: it.place ? placeBrief(it.place) : null })) });
    },
  );

  server.tool(
    "forget",
    "Forget a memory by id, or all of the user's memories for a place.",
    { memoryId: z.number().optional(), placeId: z.string().optional(), userId: z.string().optional() },
    async (a: any) => asText({ removed: mem.forget(a.userId ?? "default", { memoryId: a.memoryId, placeId: a.placeId }) }),
  );

  server.tool(
    "list_collections",
    "List the user's named place collections.",
    { userId: z.string().optional() },
    async (a: any) => asText({ collections: mem.collectionList(a.userId) }),
  );

  server.tool(
    "add_to_collection",
    "Add a place to a named collection (created on demand).",
    { name: z.string(), placeId: z.string(), userId: z.string().optional() },
    async (a: any) => { mem.collectionAdd(a.userId ?? "default", a.name, a.placeId); return asText({ added: { collection: a.name, placeId: a.placeId } }); },
  );

  await server.connect(new StdioServerTransport());
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) main();
