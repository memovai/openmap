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
import type { CandidatePlaceInput, ConversationTurn } from "./openmap.js";
import type { RankedCandidatePlace } from "./search/assist.js";
import {
  type GeoPoint,
  type Place,
  type Predicate,
  type ScoredPlace,
  personaPrefsSchema,
  relationshipSchema,
} from "./core/types.js";

type AnyOptions = Record<string, any>;

const emit = (obj: unknown) => console.log(JSON.stringify(obj, null, 2));
const list = (s?: string): string[] => (s ? s.split(",").map((x) => x.trim()).filter(Boolean) : []);

function parseNear(near?: string): GeoPoint | null {
  if (!near) return null;
  const [lat, lng] = near.split(",").map((s) => Number(s.trim()));
  if (lat == null || lng == null || Number.isNaN(lat) || Number.isNaN(lng))
    throw new Error(`invalid --near '${near}', expected 'lat,lng'`);
  return { lat, lng };
}

async function readTurns(file: string): Promise<ConversationTurn[]> {
  const turns = JSON.parse(await readFile(file, "utf-8"));
  if (!Array.isArray(turns)) throw new Error(`invalid conversation file '${file}', expected an array`);
  return turns as ConversationTurn[];
}

async function readCandidates(file: string): Promise<CandidatePlaceInput[]> {
  const candidates = JSON.parse(await readFile(file, "utf-8"));
  if (!Array.isArray(candidates)) throw new Error(`invalid candidates file '${file}', expected an array`);
  return candidates as CandidatePlaceInput[];
}

const placeBrief = (p: Place) => ({
  name: p.name, placeId: p.id, category: p.category, address: p.address,
  lat: p.lat, lng: p.lng, tags: p.tags,
});
const scored = (items: ScoredPlace[]) =>
  items.map((s) => ({
    ...placeBrief(s.place),
    score: s.score,
    distanceKm: s.distanceKm,
    relationship: s.relationship,
    source: s.place.source,
    reasons: s.reasons,
  }));
const rankedCandidates = (items: RankedCandidatePlace[]) =>
  items.map((s) => ({ ...scored([s])[0], inputIndex: s.inputIndex, memory: s.memory }));

const program = new Command();
program
  .name("openmap")
  .description("Conversation-fed place memory for AI agents. JSON to stdout.")
  .version("0.3.0")
  .option("-u, --user <id>", "user id — scopes memory, turns, beliefs, and collections", "default")
  .addHelpText("after", `
Agent loop:
  openmap context "$USER_MESSAGE"      # before answering
  openmap observe transcript.json      # after the exchange
  openmap plan "coffee near me"        # before live map search
  openmap rerank "coffee near me" candidates.json

Manual overrides and inspection live under:
  openmap manual --help
  openmap debug --help
`);

const user = (): string => program.opts().user as string;
const cmd = (parent: Command, signature: string, hidden = false): Command =>
  hidden ? parent.command(signature, { hidden: true }) : parent.command(signature);
const openMap = async () => {
  const { buildOpenMap } = await import("./openmap.js");
  return buildOpenMap();
};

// ---- action implementations ------------------------------------------------
async function observeAction(file: string, o: AnyOptions) {
  const turns = await readTurns(file);
  emit(await (await openMap()).capture(turns, { userId: user(), extract: o.extract }));
}

async function extractAction(file: string) {
  const turns = await readTurns(file);
  emit(await (await openMap()).observe(turns, { userId: user() }));
}

async function manualRememberAction(text: string, o: AnyOptions) {
  const out = await (await openMap()).remember(text, {
    userId: user(),
    relationship: relationshipSchema.parse(o.relationship ?? "mentioned"),
    note: o.note ?? null,
    near: parseNear(o.near),
    companions: list(o.companions),
  });
  emit({ stored: out.length, memories: out.map((m) => ({ id: m.id, placeId: m.placeId, relationship: m.relationship, affect: m.affect })) });
}

async function searchAction(query: string, o: AnyOptions) {
  const om = await openMap();
  const frame = await om.resolveIntent(query);
  const results = await om.recall(query, parseNear(o.near), Number(o.limit ?? 5), user());
  emit({ query, frame, results: scored(results) });
}

async function contextAction(query: string, o: AnyOptions) {
  const ctx = await (await openMap()).recallContext(query, { near: parseNear(o.near), limit: Number(o.limit ?? 5), userId: user() });
  emit({ query, system: ctx.system, prepend: ctx.prepend, places: scored(ctx.places), sources: ctx.sources });
}

async function planAction(query: string, o: AnyOptions) {
  emit(await (await openMap()).planPlaceSearch(query, { near: parseNear(o.near), userId: user() }));
}

async function rerankAction(query: string, file: string, o: AnyOptions) {
  const out = await (await openMap()).rankCandidatePlaces(query, await readCandidates(file), {
    near: parseNear(o.near),
    limit: Number(o.limit ?? 10),
    userId: user(),
  });
  emit({ query, plan: out.plan, results: rankedCandidates(out.results) });
}

async function evidenceAction(query: string, o: AnyOptions) {
  emit({ query, turns: (await openMap()).searchConversation(query, { userId: user(), limit: Number(o.limit ?? 10) }) });
}

async function intentAction(query: string) {
  emit(await (await openMap()).resolveIntent(query));
}

async function askAction(question: string) {
  emit((await openMap()).ask(question, user()));
}

async function inferAction(concept: string, o: AnyOptions) {
  emit((await openMap()).infer(concept, o.predicate as Predicate, user()));
}

async function consolidateAction() {
  emit({ written: (await openMap()).consolidate(user()) });
}

async function repairContradictionsAction() {
  emit({ repaired: (await openMap()).repairContradictions(user()) });
}

async function beliefsAction(o: AnyOptions) {
  emit({ beliefs: (await openMap()).beliefs(user(), { predicate: o.predicate, minConfidence: o.min ? Number(o.min) : undefined }) });
}

async function graphAction(o: AnyOptions) {
  const om = await openMap();
  if (o.mermaid) console.log(om.graphMermaid(user()));
  else emit(om.graph(user()));
}

async function eventsAction(o: AnyOptions) {
  emit({ events: (await openMap()).listEvents(user(), { kind: o.kind, concept: o.concept, limit: Number(o.limit ?? 50) }) });
}

async function profileAction() {
  emit((await openMap()).tasteProfile(user()));
}

async function anchorsAction() {
  emit((await openMap()).anchors(user()));
}

async function learnNearAction(km: string) {
  const om = await openMap();
  om.learn(user(), "near", Number(km));
  emit(om.anchors(user()));
}

async function calibrateAction(term: string, value: string, o: AnyOptions) {
  const om = await openMap();
  om.learn(user(), term, Number(value), o.context);
  emit({ calibrations: om.calibrations(user()) });
}

async function calibrationsAction() {
  emit({ calibrations: (await openMap()).calibrations(user()) });
}

async function regionsAction() {
  emit({ regions: (await openMap()).regions(user()) });
}

async function scenariosAction(o: AnyOptions) {
  emit({ scenarios: (await openMap()).scenarios(user(), { limit: Number(o.limit ?? 20), placeId: o.place, intent: o.intent }) });
}

async function routinesAction(o: AnyOptions) {
  emit({
    routines: (await openMap()).routines(user(), {
      limit: Number(o.limit ?? 20),
      scenarioLimit: Number(o.scenarioLimit ?? 200),
      minScenarios: Number(o.minScenarios ?? 2),
      intent: o.intent,
      concept: o.concept,
    }),
  });
}

function configAction() {
  const c = loadConfig();
  emit({
    dbPath: c.dbPath,
    embedder: resolvedEmbedder(c),
    tagger: resolvedTagger(c),
    openaiKey: Boolean(c.openaiApiKey),
    baseUrl: c.openaiBaseUrl,
    llm: c.openaiApiKey ? (c.openaiBaseUrl ? "openai-compatible (BYOC)" : "openai") : "offline (inject a runner to use a model)",
  });
}

async function serveAction() {
  const { main } = await import("./mcp.js");
  await main();
}

// ---- command builders ------------------------------------------------------
function addObserveCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <file>`, hidden)
    .description("Ingest conversation JSON: log raw turns to L0 and auto-extract memory.")
    .option("--no-extract", "only log raw turns; defer extraction")
    .action(observeAction);
}

function addContextCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <query>`, hidden)
    .description("Build memory context to inject before the agent answers.")
    .option("--near <lat,lng>")
    .option("-l, --limit <n>", "max recalled places", "5")
    .action(contextAction);
}

function addSearchCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <query>`, hidden)
    .description("Search remembered places by intent, taste, affordances, and proximity.")
    .option("--near <lat,lng>")
    .option("-l, --limit <n>", "max results", "5")
    .action(searchAction);
}

function addPlanCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <query>`, hidden)
    .description("Build memory-informed hints for a live map/place search.")
    .option("--near <lat,lng>")
    .action(planAction);
}

function addRerankCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <query> <candidates.json>`, hidden)
    .description("Personalize live place candidates from a host map provider.")
    .option("--near <lat,lng>")
    .option("-l, --limit <n>", "max results", "10")
    .action(rerankAction);
}

function addEvidenceCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <query>`, hidden)
    .description("Search raw conversation evidence from L0.")
    .option("-l, --limit <n>", "max turns", "10")
    .action(evidenceAction);
}

function addManualRememberCommand(parent: Command, name: string, hidden = false) {
  cmd(parent, `${name} <text>`, hidden)
    .description("Manual test/admin write. Normal agents should use observe instead.")
    .option("-r, --relationship <rel>", "loved|liked|visited|want_to_go|disliked|mentioned", "mentioned")
    .option("-n, --note <note>")
    .option("--near <lat,lng>")
    .option("--companions <names>")
    .action(manualRememberAction);
}

function addDebugCommands(parent: Command) {
  cmd(parent, "intent <query>")
    .description("Resolve a maps query into its latent intent frame.")
    .action(intentAction);

  cmd(parent, "ask <question>")
    .description('Infer a preference from behavior, e.g. "do I like coffee?".')
    .action(askAction);

  cmd(parent, "infer <concept>")
    .description("Infer confidence that the user likes/avoids a concept.")
    .option("-p, --predicate <p>", "likes|avoids|prefers", "likes")
    .action(inferAction);

  cmd(parent, "beliefs")
    .description("List semantic belief edges.")
    .option("-p, --predicate <p>")
    .option("--min <conf>", "minimum confidence")
    .action(beliefsAction);

  cmd(parent, "graph")
    .description("Dump the personal knowledge graph.")
    .option("--mermaid", "render as Mermaid instead of JSON")
    .action(graphAction);

  cmd(parent, "events")
    .description("List episodic events from L0/L1.")
    .option("--kind <kind>")
    .option("--concept <concept>")
    .option("-l, --limit <n>", "max", "50")
    .action(eventsAction);

  cmd(parent, "profile")
    .description("Taste profile: persona, top beliefs, favorites, stats.")
    .action(profileAction);

  cmd(parent, "anchors")
    .description("Learned spatial self-model: home, work, usual area, near radius.")
    .action(anchorsAction);

  cmd(parent, "calibrations")
    .description("Learned fuzzy-term meanings: near, walk_time, budget, noise, crowd, transit_walk.")
    .action(calibrationsAction);

  cmd(parent, "regions")
    .description("Areas the user is active in.")
    .action(regionsAction);

  cmd(parent, "scenarios")
    .description("List L2 scenario summaries.")
    .option("-l, --limit <n>", "max scenarios", "20")
    .option("--place <placeId>", "filter by place id")
    .option("--intent <intent>", "filter by intent/goal")
    .action(scenariosAction);

  cmd(parent, "routines")
    .description("List long-horizon routines derived from repeated scenarios.")
    .option("-l, --limit <n>", "max routines", "20")
    .option("--scenario-limit <n>", "max recent scenarios to roll up", "200")
    .option("--min-scenarios <n>", "minimum scenarios per routine", "2")
    .option("--intent <intent>", "filter by intent/goal")
    .option("--concept <concept>", "filter by concept")
    .action(routinesAction);

  cmd(parent, "config")
    .description("Show resolved local configuration.")
    .action(configAction);
}

function addPersonaCommands(parent: Command) {
  parent.description("Manual persona overrides.");
  cmd(parent, "show").action(async () => emit((await openMap()).getPersona(user())));
  cmd(parent, "set")
    .description("Merge explicit preference overrides. Lists are comma-separated.")
    .option("--likes <csv>").option("--dislikes <csv>").option("--vibes <csv>")
    .option("--dietary <csv>").option("--budget <level>", "low|mid|high").option("--notes <text>")
    .action(async (o) => {
      const patch = personaPrefsSchema.parse({
        ...(o.likes !== undefined ? { likes: list(o.likes) } : {}),
        ...(o.dislikes !== undefined ? { dislikes: list(o.dislikes) } : {}),
        ...(o.vibes !== undefined ? { vibes: list(o.vibes) } : {}),
        ...(o.dietary !== undefined ? { dietary: list(o.dietary) } : {}),
        ...(o.budget !== undefined ? { budget: o.budget } : {}),
        ...(o.notes !== undefined ? { notes: o.notes } : {}),
      });
      emit((await openMap()).setPersona(user(), patch));
    });
  cmd(parent, "clear").action(async () => {
    (await openMap()).clearPersona(user());
    emit({ cleared: true, user: user() });
  });
}

function addMemoryCommands(parent: Command) {
  parent.description("Manual memory management.");
  cmd(parent, "list")
    .option("-r, --relationship <rel>")
    .option("-l, --limit <n>", "max", "50")
    .action(async (o) => {
      const items = (await openMap()).listMemories(user(), {
        relationship: o.relationship ? relationshipSchema.parse(o.relationship) : undefined,
        limit: Number(o.limit ?? 50),
      });
      emit({
        count: items.length,
        memories: items.map((it) => ({
          id: it.memory.id,
          relationship: it.memory.relationship,
          affect: it.memory.affect,
          note: it.memory.note,
          sourceRefs: it.memory.sourceRefs ?? [],
          place: it.place ? placeBrief(it.place) : null,
        })),
      });
    });
  cmd(parent, "forget")
    .option("--id <memoryId>")
    .option("--place <placeId>")
    .action(async (o) => emit({ removed: (await openMap()).forget(user(), { memoryId: o.id ? Number(o.id) : undefined, placeId: o.place }) }));
  cmd(parent, "update <id>")
    .option("-r, --relationship <rel>")
    .option("-n, --note <note>")
    .action(async (id: string, o) =>
      emit({ updated: (await openMap()).updateMemory(Number(id), { relationship: o.relationship ? relationshipSchema.parse(o.relationship) : undefined, note: o.note }) }),
    );
  cmd(parent, "export").action(async () => emit((await openMap()).exportMemories(user())));
  cmd(parent, "import <file>").action(async (file: string) => emit({ imported: await (await openMap()).importMemories(JSON.parse(await readFile(file, "utf-8")), user()) }));
}

function addPlacesCommands(parent: Command) {
  parent.description("Manual place browsing and canonicalization.");
  cmd(parent, "list")
    .option("--tag <tag>")
    .option("-l, --limit <n>", "max", "50")
    .action(async (o) => {
      const ps = (await openMap()).listPlaces(user(), { tag: o.tag, limit: Number(o.limit ?? 50) });
      emit({ count: ps.length, places: ps.map(placeBrief) });
    });
  cmd(parent, "related <placeId>")
    .description("Place-to-place relations: near and similar.")
    .option("-l, --limit <n>", "max", "5")
    .action(async (placeId: string, o) => emit({ related: (await openMap()).relatedPlaces(placeId, { limit: Number(o.limit ?? 5) }) }));
  cmd(parent, "home <placeId>")
    .description("Mark a place as the user's home.")
    .action(async (placeId: string) => emit((await openMap()).setPlaceRole(user(), placeId, "home")));
  cmd(parent, "work <placeId>")
    .description("Mark a place as the user's work.")
    .action(async (placeId: string) => emit((await openMap()).setPlaceRole(user(), placeId, "work")));
  cmd(parent, "alias <alias> <placeId>")
    .description("Resolve future mentions of an alias to an existing canonical place.")
    .action(async (alias: string, placeId: string) => emit({ alias: (await openMap()).addPlaceAlias(user(), alias, placeId) }));
  cmd(parent, "aliases [placeId]")
    .description("List per-user place aliases, optionally for one canonical place.")
    .action(async (placeId?: string) => emit({ aliases: (await openMap()).placeAliases(user(), placeId) }));
}

function addCollectionCommands(parent: Command) {
  parent.description("Manual named place lists.");
  cmd(parent, "list").action(async () => emit({ collections: (await openMap()).collectionList(user()) }));
  cmd(parent, "add <name> <placeId>").action(async (name: string, placeId: string) => {
    (await openMap()).collectionAdd(user(), name, placeId);
    emit({ added: { collection: name, placeId } });
  });
  cmd(parent, "remove <name> <placeId>").action(async (name: string, placeId: string) => emit({ removed: (await openMap()).collectionRemove(user(), name, placeId) }));
  cmd(parent, "show <name>").action(async (name: string) => {
    const ps = (await openMap()).collectionShow(user(), name);
    emit({ collection: name, count: ps.length, places: ps.map(placeBrief) });
  });
}

function addManualCommands(parent: Command) {
  addManualRememberCommand(parent, "remember");
  cmd(parent, "extract <file>")
    .description("Run extraction/reconciliation without recording raw L0 turns.")
    .action(extractAction);
  cmd(parent, "learn-near <km>")
    .description("Teach the user's near tolerance from an accepted distance.")
    .action(learnNearAction);
  cmd(parent, "calibrate <term> <value>")
    .description("Teach what a fuzzy term means. term: near|walk_time|budget|noise|crowd|transit_walk")
    .option("--context <c>", "scope to a context, e.g. a goal like date")
    .action(calibrateAction);
  cmd(parent, "consolidate")
    .description("Promote behavior into persisted beliefs.")
    .action(consolidateAction);
  cmd(parent, "repair-contradictions")
    .description("Repair old inferred graph contradictions.")
    .action(repairContradictionsAction);

  addPersonaCommands(cmd(parent, "persona"));
  addMemoryCommands(cmd(parent, "memory"));
  addPlacesCommands(cmd(parent, "places"));
  addCollectionCommands(cmd(parent, "collection"));
}

// ---- primary CLI -----------------------------------------------------------
addObserveCommand(program, "observe");
addContextCommand(program, "context");
addPlanCommand(program, "plan");
addRerankCommand(program, "rerank");
addSearchCommand(program, "search");
addEvidenceCommand(program, "evidence");

const debug = cmd(program, "debug").description("Inspect derived memory state.");
addDebugCommands(debug);

const manual = cmd(program, "manual").description("Manual overrides for tests/admin. Agents should usually not use this.");
addManualCommands(manual);

cmd(program, "config")
  .description("Show resolved local configuration.")
  .action(configAction);

cmd(program, "serve")
  .description("Run the MCP server so agents can use openmap as tools.")
  .action(serveAction);

// ---- legacy compatibility aliases (hidden from help) -----------------------
addObserveCommand(program, "capture", true);
addContextCommand(program, "recall-context", true);
addSearchCommand(program, "recall", true);
addEvidenceCommand(program, "conversation", true);
addManualRememberCommand(program, "remember", true);
cmd(program, "intent <query>", true).action(intentAction);
cmd(program, "ask <question>", true).action(askAction);
cmd(program, "infer <concept>", true).option("-p, --predicate <p>", "likes|avoids|prefers", "likes").action(inferAction);
cmd(program, "consolidate", true).action(consolidateAction);
cmd(program, "repair-contradictions", true).action(repairContradictionsAction);
cmd(program, "beliefs", true).option("-p, --predicate <p>").option("--min <conf>", "minimum confidence").action(beliefsAction);
cmd(program, "graph", true).option("--mermaid", "render as Mermaid instead of JSON").action(graphAction);
cmd(program, "events", true).option("--kind <kind>").option("--concept <concept>").option("-l, --limit <n>", "max", "50").action(eventsAction);
cmd(program, "profile", true).action(profileAction);
cmd(program, "anchors", true).action(anchorsAction);
cmd(program, "learn-near <km>", true).action(learnNearAction);
cmd(program, "calibrate <term> <value>", true).option("--context <c>", "scope to a context").action(calibrateAction);
cmd(program, "calibrations", true).action(calibrationsAction);
cmd(program, "regions", true).action(regionsAction);
cmd(program, "scenarios", true)
  .option("-l, --limit <n>", "max scenarios", "20")
  .option("--place <placeId>", "filter by place id")
  .option("--intent <intent>", "filter by intent/goal")
  .action(scenariosAction);
cmd(program, "routines", true)
  .option("-l, --limit <n>", "max routines", "20")
  .option("--scenario-limit <n>", "max recent scenarios to roll up", "200")
  .option("--min-scenarios <n>", "minimum scenarios per routine", "2")
  .option("--intent <intent>", "filter by intent/goal")
  .option("--concept <concept>", "filter by concept")
  .action(routinesAction);
cmd(program, "serve-mcp", true).action(serveAction);
addPersonaCommands(cmd(program, "persona", true));
addMemoryCommands(cmd(program, "memory", true));
addPlacesCommands(cmd(program, "places", true));
addCollectionCommands(cmd(program, "collection", true));

program.parseAsync().catch((err) => {
  console.error(JSON.stringify({ error: String(err?.message ?? err) }));
  process.exit(1);
});
