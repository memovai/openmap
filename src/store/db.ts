import { DatabaseSync } from "node:sqlite";
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  type Belief,
  type Memory,
  type OmEvent,
  type Place,
  type PersonaPrefs,
  type Predicate,
  type Relationship,
  nowIso,
  placeTextBlob,
} from "../core/types.js";

/** Build an FTS5 MATCH expression from free text: OR of quoted tokens (≥2 chars). */
export function buildFtsMatch(text: string): string | null {
  const toks = [...new Set((text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length >= 2))];
  return toks.length ? toks.map((t) => `"${t}"`).join(" OR ") : null;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS places (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, lat REAL, lng REAL,
  category TEXT, address TEXT, source TEXT, source_id TEXT,
  tags TEXT, raw TEXT, embedding BLOB, dim INTEGER, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL DEFAULT 'default',
  place_id TEXT NOT NULL, relationship TEXT NOT NULL, affect REAL NOT NULL DEFAULT 0,
  note TEXT, companions TEXT, occurred_at TEXT, created_at TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_user_place ON memories(user_id, place_id);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
  kind TEXT NOT NULL, text TEXT, place_id TEXT,
  concepts TEXT, intents TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
CREATE TABLE IF NOT EXISTS beliefs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
  subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
  otype TEXT NOT NULL, confidence REAL NOT NULL, support TEXT, source TEXT, updated_at TEXT,
  UNIQUE(user_id, subject, predicate, object)
);
CREATE TABLE IF NOT EXISTS personas (
  user_id TEXT PRIMARY KEY, prefs TEXT NOT NULL, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS calibrations (
  user_id TEXT NOT NULL, term TEXT NOT NULL, value REAL, samples INTEGER NOT NULL DEFAULT 0, updated_at TEXT,
  PRIMARY KEY (user_id, term)
);
CREATE TABLE IF NOT EXISTS collections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, name TEXT NOT NULL,
  created_at TEXT, UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS collection_items (
  collection_id INTEGER NOT NULL, place_id TEXT NOT NULL, added_at TEXT,
  PRIMARY KEY (collection_id, place_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS place_fts USING fts5(place_id UNINDEXED, text);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL DEFAULT 'default',
  role TEXT NOT NULL, content TEXT NOT NULL, occurred_at TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(turn_id UNINDEXED, user_id UNINDEXED, content);
`;

export interface RememberedRow {
  place: Place;
  embedding: Float32Array | null; // null when no embedding provider is configured
  aggAffect: number;
  relationship: Relationship;
}
export interface MemoryListItem {
  memory: Memory;
  place: Place | null;
}
export interface CollectionInfo {
  id: number;
  name: string;
  count: number;
}
/** A raw conversation turn as stored in the L0 log. */
export interface StoredTurn {
  id: number;
  role: string;
  content: string;
  at: string | null;
}

export class DB {
  private db: DatabaseSync;
  /** Whether the sqlite-vec extension loaded (vec0 KNN available). */
  private vec = false;
  /** Dimension the place_vec virtual table was created with (0 = not yet). */
  private vecDim = 0;

  constructor(path: string) {
    if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
    this.db = new DatabaseSync(path, { allowExtension: true });
    this.db.exec(SCHEMA);
    const cols = this.db.prepare("PRAGMA table_info(memories)").all() as Array<{ name: string }>;
    if (!cols.some((c) => c.name === "user_id"))
      this.db.exec("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'");
    this.loadVec();
  }

  get vecEnabled(): boolean {
    return this.vec;
  }

  /** Load sqlite-vec; if unavailable the store degrades to brute-force cosine. */
  private loadVec(): void {
    try {
      const require = createRequire(import.meta.url);
      const sqliteVec = require("sqlite-vec") as { load(db: DatabaseSync): void };
      this.db.enableLoadExtension(true);
      sqliteVec.load(this.db);
      this.vec = true;
      const meta = this.db.prepare("SELECT dim FROM places WHERE dim IS NOT NULL LIMIT 1").get() as
        | { dim: number }
        | undefined;
      if (meta?.dim) this.ensureVecTable(meta.dim);
    } catch {
      this.vec = false;
    }
  }

  private ensureVecTable(dim: number): void {
    if (!this.vec || this.vecDim === dim) return;
    if (this.vecDim !== 0) this.db.exec("DROP TABLE IF EXISTS place_vec"); // dim changed (provider switch)
    this.db.exec(
      `CREATE VIRTUAL TABLE IF NOT EXISTS place_vec USING vec0(place_id TEXT PRIMARY KEY, embedding float[${dim}] distance_metric=cosine)`,
    );
    this.vecDim = dim;
  }

  /** k-NN over stored place embeddings. Uses sqlite-vec vec0 when available,
   * else brute-force cosine over the Float32 blobs. Returns cosine scores. */
  searchPlaceVectors(embedding: Float32Array, k = 10): Array<{ placeId: string; score: number }> {
    if (this.vec && this.vecDim === embedding.length) {
      const buf = Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
      const rows = this.db
        .prepare("SELECT place_id, distance FROM place_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance")
        .all(buf, k) as Array<{ place_id: string; distance: number }>;
      return rows
        .filter((r) => r.distance != null && !Number.isNaN(r.distance))
        .map((r) => ({ placeId: r.place_id, score: 1 - r.distance }));
    }
    // fallback: brute-force over blobs (embeddings are L2-normalized → dot = cosine)
    const scored = this.allPlacesWithEmbeddings()
      .filter((x) => x.embedding && x.embedding.length === embedding.length)
      .map((x) => {
        let s = 0;
        for (let i = 0; i < embedding.length; i++) s += embedding[i]! * x.embedding![i]!;
        return { placeId: x.place.id, score: s };
      });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  /** Keyword (FTS5/BM25) search over place text. Returns place_ids best-first.
   * No embedding needed — this is the always-available arm of hybrid recall. */
  searchPlaceFts(matchQuery: string, k = 20): string[] {
    try {
      const rows = this.db
        .prepare("SELECT place_id FROM place_fts WHERE place_fts MATCH ? ORDER BY rank LIMIT ?")
        .all(matchQuery, k) as Array<{ place_id: string }>;
      return rows.map((r) => r.place_id);
    } catch {
      return []; // malformed MATCH / FTS unavailable → keyword arm empty
    }
  }

  // ---- L0 raw conversation log -------------------------------------------
  /** Append raw turns verbatim, so the original wording can be recalled later
   * for grounding ("when did I say I loved X"). Returns rows inserted. */
  recordTurns(userId: string, turns: Array<{ role: string; content: string; at?: string }>, now: string): number {
    const ins = this.db.prepare(
      "INSERT INTO turns (user_id, role, content, occurred_at, created_at) VALUES (?, ?, ?, ?, ?)",
    );
    const fts = this.db.prepare("INSERT INTO turn_fts (turn_id, user_id, content) VALUES (?, ?, ?)");
    let n = 0;
    for (const t of turns) {
      const content = (t.content ?? "").trim();
      if (!content) continue;
      const r = ins.run(userId, t.role || "user", content, t.at ?? now, now);
      fts.run(Number(r.lastInsertRowid), userId, content);
      n++;
    }
    return n;
  }

  /** BM25 keyword search over raw turns, newest-first on ties. */
  searchTurns(userId: string, matchQuery: string, k = 10): StoredTurn[] {
    try {
      const rows = this.db
        .prepare(
          "SELECT t.id, t.role, t.content, t.occurred_at FROM turn_fts f " +
            "JOIN turns t ON t.id = f.turn_id WHERE f.user_id = ? AND turn_fts MATCH ? " +
            "ORDER BY rank LIMIT ?",
        )
        .all(userId, matchQuery, k) as Array<{ id: number; role: string; content: string; occurred_at: string | null }>;
      return rows.map((r) => ({ id: r.id, role: r.role, content: r.content, at: r.occurred_at }));
    } catch {
      return [];
    }
  }

  recentTurns(userId: string, k = 20): StoredTurn[] {
    const rows = this.db
      .prepare("SELECT id, role, content, occurred_at FROM turns WHERE user_id = ? ORDER BY id DESC LIMIT ?")
      .all(userId, k) as Array<{ id: number; role: string; content: string; occurred_at: string | null }>;
    return rows.map((r) => ({ id: r.id, role: r.role, content: r.content, at: r.occurred_at })).reverse();
  }

  countTurns(userId: string): number {
    const r = this.db.prepare("SELECT COUNT(*) AS n FROM turns WHERE user_id = ?").get(userId) as { n: number };
    return r.n;
  }

  // ---- places -------------------------------------------------------------
  upsertPlace(place: Place, embedding?: Float32Array | null): void {
    const blob = embedding ? f32ToBlob(embedding) : null;
    const dim = embedding ? embedding.length : null;
    // keep the FTS index in sync (no UPSERT on fts5 → delete + insert)
    this.db.prepare("DELETE FROM place_fts WHERE place_id=?").run(place.id);
    this.db.prepare("INSERT INTO place_fts (place_id, text) VALUES (?, ?)").run(place.id, placeTextBlob(place));
    if (embedding && this.vec) {
      this.ensureVecTable(embedding.length);
      if (this.vecDim === embedding.length) {
        const buf = Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
        this.db.prepare("DELETE FROM place_vec WHERE place_id=?").run(place.id); // vec0 has no upsert
        this.db.prepare("INSERT INTO place_vec (place_id, embedding) VALUES (?, ?)").run(place.id, buf);
      }
    }
    if (!this.getPlace(place.id)) {
      this.db
        .prepare(
          `INSERT INTO places (id,name,lat,lng,category,address,source,source_id,tags,raw,embedding,dim,created_at,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
        )
        .run(place.id, place.name, place.lat, place.lng, place.category, place.address, place.source,
          place.sourceId, JSON.stringify(place.tags), JSON.stringify(place.raw), blob, dim, nowIso(), nowIso());
      return;
    }
    const keep = blob ?? this.rawEmbedding(place.id)?.blob ?? null;
    const keepDim = blob ? dim : (this.rawEmbedding(place.id)?.dim ?? null);
    this.db
      .prepare(
        `UPDATE places SET name=?,lat=?,lng=?,category=?,address=?,source=?,source_id=?,tags=?,raw=?,embedding=?,dim=?,updated_at=? WHERE id=?`,
      )
      .run(place.name, place.lat, place.lng, place.category, place.address, place.source, place.sourceId,
        JSON.stringify(place.tags), JSON.stringify(place.raw), keep, keepDim, nowIso(), place.id);
  }

  private rawEmbedding(id: string): { blob: Uint8Array; dim: number } | null {
    const row = this.db.prepare("SELECT embedding, dim FROM places WHERE id=?").get(id) as
      | { embedding: Uint8Array | null; dim: number | null }
      | undefined;
    if (!row?.embedding || row.dim == null) return null;
    return { blob: row.embedding, dim: row.dim };
  }

  getPlace(id: string): Place | null {
    const row = this.db.prepare("SELECT * FROM places WHERE id=?").get(id) as PlaceRow | undefined;
    return row ? rowToPlace(row) : null;
  }

  embeddingFor(id: string): Float32Array | null {
    const r = this.rawEmbedding(id);
    return r ? blobToF32(r.blob) : null;
  }

  listPlaces(userId: string, opts: { tag?: string; limit?: number } = {}): Place[] {
    const params: unknown[] = [userId];
    let sql = `SELECT DISTINCT p.* FROM places p JOIN memories m ON m.place_id=p.id WHERE m.user_id=?`;
    if (opts.tag) {
      sql += ` AND p.tags LIKE ?`;
      params.push(`%"${opts.tag}"%`);
    }
    sql += ` ORDER BY p.updated_at DESC LIMIT ?`;
    params.push(opts.limit ?? 50);
    return (this.db.prepare(sql).all(...(params as any[])) as unknown as PlaceRow[]).map(rowToPlace);
  }

  allPlacesWithEmbeddings(limit = 5000): Array<{ place: Place; embedding: Float32Array | null }> {
    const rows = this.db.prepare("SELECT * FROM places LIMIT ?").all(limit) as unknown as PlaceRow[];
    return rows.map((r) => ({ place: rowToPlace(r), embedding: r.embedding ? blobToF32(r.embedding) : null }));
  }

  // ---- memories -----------------------------------------------------------
  addMemory(mem: Memory): number {
    const res = this.db
      .prepare(
        `INSERT INTO memories (user_id,place_id,relationship,affect,note,companions,occurred_at,created_at,source)
         VALUES (?,?,?,?,?,?,?,?,?)`,
      )
      .run(mem.userId, mem.placeId, mem.relationship, mem.affect, mem.note,
        JSON.stringify(mem.companions), mem.occurredAt, mem.createdAt, mem.source);
    return Number(res.lastInsertRowid);
  }

  memoriesFor(placeId: string, userId: string): Memory[] {
    return (
      this.db.prepare("SELECT * FROM memories WHERE place_id=? AND user_id=? ORDER BY created_at").all(placeId, userId) as unknown as MemoryRow[]
    ).map(rowToMemory);
  }

  listMemories(userId: string, opts: { relationship?: Relationship; limit?: number } = {}): MemoryListItem[] {
    const params: unknown[] = [userId];
    let sql = "SELECT * FROM memories WHERE user_id=?";
    if (opts.relationship) {
      sql += " AND relationship=?";
      params.push(opts.relationship);
    }
    sql += " ORDER BY created_at DESC LIMIT ?";
    params.push(opts.limit ?? 50);
    const rows = this.db.prepare(sql).all(...(params as any[])) as unknown as MemoryRow[];
    return rows.map((r) => {
      const mem = rowToMemory(r);
      return { memory: mem, place: this.getPlace(mem.placeId) };
    });
  }

  getMemory(id: number): Memory | null {
    const row = this.db.prepare("SELECT * FROM memories WHERE id=?").get(id) as MemoryRow | undefined;
    return row ? rowToMemory(row) : null;
  }

  updateMemory(id: number, f: { relationship?: Relationship; affect?: number; note?: string | null }): boolean {
    const sets: string[] = [];
    const params: unknown[] = [];
    if (f.relationship !== undefined) (sets.push("relationship=?"), params.push(f.relationship));
    if (f.affect !== undefined) (sets.push("affect=?"), params.push(f.affect));
    if (f.note !== undefined) (sets.push("note=?"), params.push(f.note));
    if (sets.length === 0) return false;
    params.push(id);
    return Number(this.db.prepare(`UPDATE memories SET ${sets.join(",")} WHERE id=?`).run(...(params as any[])).changes) > 0;
  }

  deleteMemory(id: number, userId: string): number {
    return Number(this.db.prepare("DELETE FROM memories WHERE id=? AND user_id=?").run(id, userId).changes);
  }

  forgetPlace(userId: string, placeId: string): number {
    return Number(this.db.prepare("DELETE FROM memories WHERE user_id=? AND place_id=?").run(userId, placeId).changes);
  }

  iterRemembered(userId: string): RememberedRow[] {
    const rows = this.db
      .prepare(
        `SELECT p.*, (SELECT AVG(affect) FROM memories m WHERE m.place_id=p.id AND m.user_id=$u) AS agg_affect
         FROM places p WHERE EXISTS (SELECT 1 FROM memories m WHERE m.place_id=p.id AND m.user_id=$u)`,
      )
      .all({ u: userId } as any) as unknown as (PlaceRow & { agg_affect: number | null })[];
    const out: RememberedRow[] = [];
    for (const row of rows) {
      out.push({
        place: rowToPlace(row),
        embedding: row.embedding ? blobToF32(row.embedding) : null, // null when no embedder
        aggAffect: row.agg_affect ?? 0,
        relationship: this.strongestRelationship(row.id, userId),
      });
    }
    return out;
  }

  private strongestRelationship(placeId: string, userId: string): Relationship {
    const rows = this.memoriesFor(placeId, userId);
    if (rows.length === 0) return "mentioned";
    return rows.reduce((a, b) => (b.affect > a.affect ? b : a)).relationship;
  }

  // ---- events (L0) --------------------------------------------------------
  addEvent(e: OmEvent): number {
    const res = this.db
      .prepare(`INSERT INTO events (user_id,kind,text,place_id,concepts,intents,created_at) VALUES (?,?,?,?,?,?,?)`)
      .run(e.userId, e.kind, e.text, e.placeId, JSON.stringify(e.concepts), JSON.stringify(e.intents), e.createdAt);
    return Number(res.lastInsertRowid);
  }

  listEvents(userId: string, opts: { kind?: string; concept?: string; limit?: number } = {}): OmEvent[] {
    const params: unknown[] = [userId];
    let sql = "SELECT * FROM events WHERE user_id=?";
    if (opts.kind) {
      sql += " AND kind=?";
      params.push(opts.kind);
    }
    if (opts.concept) {
      sql += " AND concepts LIKE ?";
      params.push(`%"${opts.concept}"%`);
    }
    sql += " ORDER BY created_at DESC LIMIT ?";
    params.push(opts.limit ?? 100);
    return (this.db.prepare(sql).all(...(params as any[])) as unknown as EventRow[]).map(rowToEvent);
  }

  // ---- beliefs (L2) -------------------------------------------------------
  upsertBelief(b: Belief): void {
    this.db
      .prepare(
        `INSERT INTO beliefs (user_id,subject,predicate,object,otype,confidence,support,source,updated_at)
         VALUES (?,?,?,?,?,?,?,?,?)
         ON CONFLICT(user_id,subject,predicate,object) DO UPDATE SET
           confidence=excluded.confidence, support=excluded.support,
           source=excluded.source, otype=excluded.otype, updated_at=excluded.updated_at`,
      )
      .run(b.userId, b.subject, b.predicate, b.object, b.otype, b.confidence,
        JSON.stringify(b.support), b.source, b.updatedAt);
  }

  getBelief(userId: string, subject: string, predicate: Predicate, object: string): Belief | null {
    const row = this.db
      .prepare("SELECT * FROM beliefs WHERE user_id=? AND subject=? AND predicate=? AND object=?")
      .get(userId, subject, predicate, object) as BeliefRow | undefined;
    return row ? rowToBelief(row) : null;
  }

  listBeliefs(userId: string, opts: { predicate?: Predicate; minConfidence?: number } = {}): Belief[] {
    const params: unknown[] = [userId];
    let sql = "SELECT * FROM beliefs WHERE user_id=?";
    if (opts.predicate) {
      sql += " AND predicate=?";
      params.push(opts.predicate);
    }
    if (opts.minConfidence != null) {
      sql += " AND confidence>=?";
      params.push(opts.minConfidence);
    }
    sql += " ORDER BY confidence DESC";
    return (this.db.prepare(sql).all(...(params as any[])) as unknown as BeliefRow[]).map(rowToBelief);
  }

  deleteBelief(userId: string, id: number): number {
    return Number(this.db.prepare("DELETE FROM beliefs WHERE user_id=? AND id=?").run(userId, id).changes);
  }

  // ---- persona ------------------------------------------------------------
  getPersonaPrefs(userId: string): { prefs: PersonaPrefs | null; updatedAt: string | null } {
    const row = this.db.prepare("SELECT * FROM personas WHERE user_id=?").get(userId) as
      | { prefs: string; updated_at: string | null }
      | undefined;
    return row ? { prefs: JSON.parse(row.prefs) as PersonaPrefs, updatedAt: row.updated_at } : { prefs: null, updatedAt: null };
  }

  setPersonaPrefs(userId: string, prefs: PersonaPrefs): void {
    this.db
      .prepare(
        `INSERT INTO personas (user_id, prefs, updated_at) VALUES (?,?,?)
         ON CONFLICT(user_id) DO UPDATE SET prefs=excluded.prefs, updated_at=excluded.updated_at`,
      )
      .run(userId, JSON.stringify(prefs), nowIso());
  }

  deletePersona(userId: string): void {
    this.db.prepare("DELETE FROM personas WHERE user_id=?").run(userId);
  }

  // ---- collections --------------------------------------------------------
  collectionUpsert(userId: string, name: string): number {
    const e = this.collectionByName(userId, name);
    if (e) return e.id;
    return Number(this.db.prepare("INSERT INTO collections (user_id,name,created_at) VALUES (?,?,?)").run(userId, name, nowIso()).lastInsertRowid);
  }
  collectionByName(userId: string, name: string): { id: number; name: string } | null {
    return (this.db.prepare("SELECT id,name FROM collections WHERE user_id=? AND name=?").get(userId, name) as { id: number; name: string } | undefined) ?? null;
  }
  collectionList(userId: string): CollectionInfo[] {
    return this.db
      .prepare(
        `SELECT c.id,c.name,(SELECT COUNT(*) FROM collection_items i WHERE i.collection_id=c.id) AS count
         FROM collections c WHERE c.user_id=? ORDER BY c.name`,
      )
      .all(userId) as unknown as CollectionInfo[];
  }
  collectionAddItem(collectionId: number, placeId: string): void {
    this.db.prepare(`INSERT INTO collection_items (collection_id,place_id,added_at) VALUES (?,?,?) ON CONFLICT(collection_id,place_id) DO NOTHING`).run(collectionId, placeId, nowIso());
  }
  collectionRemoveItem(collectionId: number, placeId: string): number {
    return Number(this.db.prepare("DELETE FROM collection_items WHERE collection_id=? AND place_id=?").run(collectionId, placeId).changes);
  }
  collectionItems(collectionId: number): Place[] {
    return (this.db.prepare(`SELECT p.* FROM places p JOIN collection_items i ON i.place_id=p.id WHERE i.collection_id=? ORDER BY i.added_at DESC`).all(collectionId) as unknown as PlaceRow[]).map(rowToPlace);
  }

  // ---- calibration store (learned per-user meaning of fuzzy terms) --------
  getCalibration(userId: string, term: string): { value: number | null; samples: number } | null {
    const row = this.db.prepare("SELECT value, samples FROM calibrations WHERE user_id=? AND term=?").get(userId, term) as
      | { value: number | null; samples: number }
      | undefined;
    return row ? { value: row.value, samples: row.samples } : null;
  }

  upsertCalibration(userId: string, term: string, value: number, samples: number): void {
    this.db
      .prepare(
        `INSERT INTO calibrations (user_id, term, value, samples, updated_at) VALUES (?,?,?,?,?)
         ON CONFLICT(user_id, term) DO UPDATE SET value=excluded.value, samples=excluded.samples, updated_at=excluded.updated_at`,
      )
      .run(userId, term, value, samples, nowIso());
  }

  listCalibrations(userId: string): Array<{ term: string; value: number | null; samples: number }> {
    return this.db
      .prepare("SELECT term, value, samples FROM calibrations WHERE user_id=? ORDER BY term")
      .all(userId) as unknown as Array<{ term: string; value: number | null; samples: number }>;
  }

  stats(userId?: string): { places: number; memories: number; events: number; beliefs: number } {
    if (userId) {
      const c = (sql: string) => (this.db.prepare(sql).get(userId) as { c: number }).c;
      return {
        places: c("SELECT COUNT(DISTINCT place_id) c FROM memories WHERE user_id=?"),
        memories: c("SELECT COUNT(*) c FROM memories WHERE user_id=?"),
        events: c("SELECT COUNT(*) c FROM events WHERE user_id=?"),
        beliefs: c("SELECT COUNT(*) c FROM beliefs WHERE user_id=?"),
      };
    }
    const g = (sql: string) => (this.db.prepare(sql).get() as { c: number }).c;
    return {
      places: g("SELECT COUNT(*) c FROM places"),
      memories: g("SELECT COUNT(*) c FROM memories"),
      events: g("SELECT COUNT(*) c FROM events"),
      beliefs: g("SELECT COUNT(*) c FROM beliefs"),
    };
  }

  close(): void {
    this.db.close();
  }
}

// ---- row mapping ----------------------------------------------------------
interface PlaceRow {
  id: string; name: string; lat: number | null; lng: number | null; category: string | null;
  address: string | null; source: string | null; source_id: string | null; tags: string | null;
  raw: string | null; embedding: Uint8Array | null; dim: number | null;
}
interface MemoryRow {
  id: number; user_id: string | null; place_id: string; relationship: string; affect: number;
  note: string | null; companions: string | null; occurred_at: string | null; created_at: string | null; source: string | null;
}
interface EventRow {
  id: number; user_id: string; kind: string; text: string | null; place_id: string | null;
  concepts: string | null; intents: string | null; created_at: string | null;
}
interface BeliefRow {
  id: number; user_id: string; subject: string; predicate: string; object: string; otype: string;
  confidence: number; support: string | null; source: string | null; updated_at: string | null;
}

function rowToPlace(r: PlaceRow): Place {
  return {
    id: r.id, name: r.name, lat: r.lat, lng: r.lng, category: r.category, address: r.address,
    source: r.source ?? "unknown", sourceId: r.source_id,
    tags: r.tags ? (JSON.parse(r.tags) as string[]) : [],
    raw: r.raw ? (JSON.parse(r.raw) as Record<string, unknown>) : {},
  };
}
function rowToMemory(r: MemoryRow): Memory {
  return {
    id: r.id, userId: r.user_id ?? "default", placeId: r.place_id, relationship: r.relationship as Relationship,
    affect: r.affect, note: r.note, companions: r.companions ? (JSON.parse(r.companions) as string[]) : [],
    occurredAt: r.occurred_at, createdAt: r.created_at ?? nowIso(), source: r.source ?? "manual",
  };
}
function rowToEvent(r: EventRow): OmEvent {
  return {
    id: r.id, userId: r.user_id, kind: r.kind as OmEvent["kind"], text: r.text ?? "", placeId: r.place_id,
    concepts: r.concepts ? (JSON.parse(r.concepts) as string[]) : [],
    intents: r.intents ? (JSON.parse(r.intents) as string[]) : [],
    createdAt: r.created_at ?? nowIso(),
  };
}
function rowToBelief(r: BeliefRow): Belief {
  return {
    id: r.id, userId: r.user_id, subject: r.subject, predicate: r.predicate as Belief["predicate"],
    object: r.object, otype: r.otype as Belief["otype"], confidence: r.confidence,
    support: r.support ? (JSON.parse(r.support) as string[]) : [], source: (r.source ?? "inferred") as Belief["source"],
    updatedAt: r.updated_at ?? nowIso(),
  };
}

function f32ToBlob(v: Float32Array): Uint8Array {
  return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
}
function blobToF32(bytes: Uint8Array): Float32Array {
  const copy = bytes.slice();
  return new Float32Array(copy.buffer, 0, Math.floor(copy.byteLength / 4));
}
