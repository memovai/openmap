#!/usr/bin/env node
// Quiet the node:sqlite ExperimentalWarning so JSON stdout stays clean.
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile } from "node:fs/promises";
import { Command } from "commander";
import { loadConfig, resolvedEmbedder, resolvedTagger } from "./core/config.js";
import { buildOpenMap } from "./openmap.js";
import {
  type GeoPoint,
  type Place,
  type Predicate,
  type ScoredPlace,
  personaPrefsSchema,
  relationshipSchema,
} from "./core/types.js";

const emit = (obj: unknown) => console.log(JSON.stringify(obj, null, 2));
const list = (s?: string): string[] => (s ? s.split(",").map((x) => x.trim()).filter(Boolean) : []);

function parseNear(near?: string): GeoPoint | null {
  if (!near) return null;
  const [lat, lng] = near.split(",").map((s) => Number(s.trim()));
  if (lat == null || lng == null || Number.isNaN(lat) || Number.isNaN(lng))
    throw new Error(`invalid --near '${near}', expected 'lat,lng'`);
  return { lat, lng };
}

const placeBrief = (p: Place) => ({
  name: p.name, placeId: p.id, category: p.category, address: p.address,
  lat: p.lat, lng: p.lng, tags: p.tags,
});
const scored = (items: ScoredPlace[]) =>
  items.map((s) => ({ ...placeBrief(s.place), score: s.score, distanceKm: s.distanceKm, relationship: s.relationship, source: s.place.source, reasons: s.reasons }));

const program = new Command();
program
  .name("openmap")
  .description("A map-aware memory layer for AI agents — fed by conversation. JSON to stdout for agents.")
  .version("0.3.0")
  .option("-u, --user <id>", "user id — scopes events, memory, beliefs, persona, collections", "default");
const user = (): string => program.opts().user as string;

// ---- search / write --------------------------------------------------------
program
  .command("remember <text>")
  .description("Extract place mentions, resolve them, store memories + an event")
  .option("-r, --relationship <rel>", "loved|liked|visited|want_to_go|disliked|mentioned", "mentioned")
  .option("-n, --note <note>")
  .option("--near <lat,lng>")
  .option("--companions <names>")
  .action(async (text: string, o) => {
    const out = await buildOpenMap().remember(text, {
      userId: user(), relationship: relationshipSchema.parse(o.relationship),
      note: o.note ?? null, near: parseNear(o.near), companions: list(o.companions),
    });
    emit({ stored: out.length, memories: out.map((m) => ({ id: m.id, placeId: m.placeId, relationship: m.relationship, affect: m.affect })) });
  });

program
  .command("recall <query>")
  .description("Search your remembered places by resolved intent (taste + vibe + proximity)")
  .option("--near <lat,lng>")
  .option("-l, --limit <n>", "max results", "5")
  .action(async (query: string, o) => {
    const om = buildOpenMap();
    const frame = await om.resolveIntent(query);
    const results = await om.recall(query, parseNear(o.near), Number(o.limit), user());
    emit({ query, frame, results: scored(results) });
  });

program
  .command("intent <query>")
  .description("Resolve a maps query into its latent situational intent frame")
  .action(async (query: string) => emit(await buildOpenMap().resolveIntent(query)));

program
  .command("observe <file>")
  .description("Auto-capture memory from a conversation JSON: [{role,content}, …]. Extracts + reconciles place mentions.")
  .action(async (file: string) => {
    const turns = JSON.parse(await readFile(file, "utf-8"));
    emit(await buildOpenMap().observe(turns, { userId: user() }));
  });

// ---- agent hooks (auto-capture / auto-recall) ------------------------------
program
  .command("capture <file>")
  .description("auto-capture: log raw turns to L0 (for grounding) + extract memory. Conversation JSON: [{role,content}, …].")
  .option("--no-extract", "only log raw turns (defer LLM extraction to a cadence)")
  .action(async (file: string, o) => {
    const turns = JSON.parse(await readFile(file, "utf-8"));
    emit(await buildOpenMap().capture(turns, { userId: user(), extract: o.extract }));
  });

program
  .command("recall-context <query>")
  .description("auto-recall: build injectable context — persona block (system) + relevant places (prepend). For agent turn loops.")
  .option("--near <lat,lng>")
  .option("-l, --limit <n>", "max recalled places", "5")
  .action(async (query: string, o) => {
    const ctx = await buildOpenMap().recallContext(query, { near: parseNear(o.near), limit: Number(o.limit), userId: user() });
    emit({ query, system: ctx.system, prepend: ctx.prepend, places: scored(ctx.places) });
  });

program
  .command("conversation <query>")
  .description("Search the raw L0 conversation log (BM25) — recall original wording to ground a memory.")
  .option("-l, --limit <n>", "max turns", "10")
  .action((query: string, o) => emit({ query, turns: buildOpenMap().searchConversation(query, { userId: user(), limit: Number(o.limit) }) }));

// ---- inference -------------------------------------------------------------
program
  .command("ask <question>")
  .description('Infer a preference from behavior, e.g. "do I like coffee?"')
  .action((question: string) => emit(buildOpenMap().ask(question, user())));

program
  .command("infer <concept>")
  .description("Infer confidence that the user likes/avoids a concept")
  .option("-p, --predicate <p>", "likes|avoids|prefers", "likes")
  .action((concept: string, o) => emit(buildOpenMap().infer(concept, o.predicate as Predicate, user())));

program
  .command("consolidate")
  .description("Promote behavior (events) into persisted beliefs (L0/L1 → L2)")
  .action(() => emit({ written: buildOpenMap().consolidate(user()) }));

program
  .command("beliefs")
  .description("List the user's semantic beliefs (the personal knowledge graph edges)")
  .option("-p, --predicate <p>")
  .option("--min <conf>", "minimum confidence")
  .action((o) => emit({ beliefs: buildOpenMap().beliefs(user(), { predicate: o.predicate, minConfidence: o.min ? Number(o.min) : undefined }) }));

program
  .command("graph")
  .description("Dump the user's personal knowledge graph (nodes + edges, or Mermaid)")
  .option("--mermaid", "render as a Mermaid diagram instead of JSON")
  .action((o) => {
    const om = buildOpenMap();
    if (o.mermaid) console.log(om.graphMermaid(user()));
    else emit(om.graph(user()));
  });

program
  .command("events")
  .description("List the user's episodic event log (L0)")
  .option("--kind <kind>")
  .option("--concept <concept>")
  .option("-l, --limit <n>", "max", "50")
  .action((o) => emit({ events: buildOpenMap().listEvents(user(), { kind: o.kind, concept: o.concept, limit: Number(o.limit) }) }));

program
  .command("profile")
  .description("Taste profile: persona, top beliefs, favorites, stats")
  .action(() => emit(buildOpenMap().tasteProfile(user())));

program
  .command("anchors")
  .description("The user's learned spatial self-model (home / work / usual area / 'near' radius)")
  .action(() => emit(buildOpenMap().anchors(user())));

program
  .command("learn-near <km>")
  .description("Teach the user's 'near' tolerance from an accepted distance (e.g. picked a place 3km away)")
  .action((km: string) => {
    const om = buildOpenMap();
    om.learn(user(), "near", Number(km));
    emit(om.anchors(user()));
  });

program
  .command("calibrate <term> <value>")
  .description("Teach what a fuzzy term means to this user. term: near|walk_time|budget|noise")
  .option("--context <c>", "scope to a context, e.g. a goal like 'date' (near-for-a-date ≠ near-for-coffee)")
  .action((term: string, value: string, o) => {
    const om = buildOpenMap();
    om.learn(user(), term, Number(value), o.context);
    emit({ calibrations: om.calibrations(user()) });
  });

program
  .command("calibrations")
  .description("Show the user's learned meaning of fuzzy terms (near/walk_time/budget/noise)")
  .action(() => emit({ calibrations: buildOpenMap().calibrations(user()) }));

program
  .command("regions")
  .description("Areas the user is active in (user↔area relationship), clustered from place activity")
  .action(() => emit({ regions: buildOpenMap().regions(user()) }));

// ---- persona ---------------------------------------------------------------
const persona = program.command("persona").description("Manage explicit preferences");
persona.command("show").action(() => emit(buildOpenMap().getPersona(user())));
persona
  .command("set")
  .description("Merge preferences (provided fields replace). Lists are comma-separated.")
  .option("--likes <csv>").option("--dislikes <csv>").option("--vibes <csv>")
  .option("--dietary <csv>").option("--budget <level>", "low|mid|high").option("--notes <text>")
  .action((o) => {
    const patch = personaPrefsSchema.parse({
      ...(o.likes !== undefined ? { likes: list(o.likes) } : {}),
      ...(o.dislikes !== undefined ? { dislikes: list(o.dislikes) } : {}),
      ...(o.vibes !== undefined ? { vibes: list(o.vibes) } : {}),
      ...(o.dietary !== undefined ? { dietary: list(o.dietary) } : {}),
      ...(o.budget !== undefined ? { budget: o.budget } : {}),
      ...(o.notes !== undefined ? { notes: o.notes } : {}),
    });
    emit(buildOpenMap().setPersona(user(), patch));
  });
persona.command("clear").action(() => { buildOpenMap().clearPersona(user()); emit({ cleared: true, user: user() }); });

// ---- memory management -----------------------------------------------------
const memory = program.command("memory").description("Manage stored memories");
memory
  .command("list").option("-r, --relationship <rel>").option("-l, --limit <n>", "max", "50")
  .action((o) => {
    const items = buildOpenMap().listMemories(user(), { relationship: o.relationship ? relationshipSchema.parse(o.relationship) : undefined, limit: Number(o.limit) });
    emit({ count: items.length, memories: items.map((it) => ({ id: it.memory.id, relationship: it.memory.relationship, affect: it.memory.affect, note: it.memory.note, place: it.place ? placeBrief(it.place) : null })) });
  });
memory
  .command("forget").option("--id <memoryId>").option("--place <placeId>")
  .action((o) => emit({ removed: buildOpenMap().forget(user(), { memoryId: o.id ? Number(o.id) : undefined, placeId: o.place }) }));
memory
  .command("update <id>").option("-r, --relationship <rel>").option("-n, --note <note>")
  .action((id: string, o) => emit({ updated: buildOpenMap().updateMemory(Number(id), { relationship: o.relationship ? relationshipSchema.parse(o.relationship) : undefined, note: o.note }) }));
memory.command("export").action(() => emit(buildOpenMap().exportMemories(user())));
memory.command("import <file>").action(async (file: string) => emit({ imported: await buildOpenMap().importMemories(JSON.parse(await readFile(file, "utf-8")), user()) }));

// ---- places + collections --------------------------------------------------
const places = program.command("places").description("Browse places + place relations");
places.command("list").option("--tag <tag>").option("-l, --limit <n>", "max", "50")
  .action((o) => { const ps = buildOpenMap().listPlaces(user(), { tag: o.tag, limit: Number(o.limit) }); emit({ count: ps.length, places: ps.map(placeBrief) }); });
places.command("related <placeId>").description("Place↔place relations (near + similar)").option("-l, --limit <n>", "max", "5")
  .action((placeId: string, o) => emit({ related: buildOpenMap().relatedPlaces(placeId, { limit: Number(o.limit) }) }));
places.command("home <placeId>").description("Mark a place as the user's home (lives_near)").action((placeId: string) => emit(buildOpenMap().setPlaceRole(user(), placeId, "home")));
places.command("work <placeId>").description("Mark a place as the user's work (works_near)").action((placeId: string) => emit(buildOpenMap().setPlaceRole(user(), placeId, "work")));

const collection = program.command("collection").description("Named place lists / saved searches");
collection.command("list").action(() => emit({ collections: buildOpenMap().collectionList(user()) }));
collection.command("add <name> <placeId>").action((name: string, placeId: string) => { buildOpenMap().collectionAdd(user(), name, placeId); emit({ added: { collection: name, placeId } }); });
collection.command("remove <name> <placeId>").action((name: string, placeId: string) => emit({ removed: buildOpenMap().collectionRemove(user(), name, placeId) }));
collection.command("show <name>").action((name: string) => { const ps = buildOpenMap().collectionShow(user(), name); emit({ collection: name, count: ps.length, places: ps.map(placeBrief) }); });

// ---- misc ------------------------------------------------------------------
program
  .command("config")
  .action(() => {
    const c = loadConfig();
    emit({
      dbPath: c.dbPath, embedder: resolvedEmbedder(c), tagger: resolvedTagger(c),
      openaiKey: Boolean(c.openaiApiKey), baseUrl: c.openaiBaseUrl,
      llm: c.openaiApiKey ? (c.openaiBaseUrl ? "openai-compatible (BYOC)" : "openai") : "offline (inject a runner to use a model)",
    });
  });

program
  .command("serve-mcp")
  .description("Run the MCP server so agents can use openmap as tools")
  .action(async () => { const { main } = await import("./mcp.js"); await main(); });

program.parseAsync().catch((err) => {
  console.error(JSON.stringify({ error: String(err?.message ?? err) }));
  process.exit(1);
});
