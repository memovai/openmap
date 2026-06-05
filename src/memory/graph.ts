import { type DB } from "../store/db.js";
import { type GraphEdge, type GraphNode } from "../core/types.js";

export interface KnowledgeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** The user's personal knowledge graph: user→concept/place/goal belief edges
 * plus user→place memory edges. */
export function buildGraph(db: DB, userId: string): KnowledgeGraph {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  nodes.set("user", { id: "user", type: "user", label: "user" });
  const placeName = (id: string) => db.getPlace(id)?.name ?? id;

  for (const b of db.listBeliefs(userId)) {
    const nodeId = b.otype === "place" ? b.object : `${b.otype}:${b.object}`;
    const label = b.otype === "place" ? placeName(b.object) : b.object;
    if (!nodes.has(nodeId)) nodes.set(nodeId, { id: nodeId, type: b.otype, label });
    edges.push({ source: "user", predicate: b.predicate, target: nodeId, confidence: b.confidence });
  }
  for (const it of db.listMemories(userId, { limit: 1000 })) {
    const id = it.memory.placeId;
    if (!nodes.has(id)) nodes.set(id, { id, type: "place", label: it.place?.name ?? id });
    edges.push({ source: "user", predicate: it.memory.relationship, target: id });
  }
  return { nodes: [...nodes.values()], edges };
}

/** Render a knowledge graph as a compact, human-readable, LLM-pasteable Mermaid
 * diagram (the "symbolic memory" representation). */
export function graphToMermaid(g: KnowledgeGraph): string {
  const sid = (id: string) => "n_" + id.replace(/[^a-zA-Z0-9]/g, "_");
  const esc = (s: string) => s.replace(/"/g, "'");
  const lines = ["graph LR"];
  for (const n of g.nodes)
    lines.push(n.type === "user" ? `  ${sid(n.id)}(("${esc(n.label)}"))` : `  ${sid(n.id)}["${esc(n.label)}"]`);
  for (const e of g.edges) {
    const lbl = e.confidence != null ? `${e.predicate} ${e.confidence.toFixed(2)}` : e.predicate;
    lines.push(`  ${sid(e.source)} -->|"${esc(lbl)}"| ${sid(e.target)}`);
  }
  return lines.join("\n");
}
