import type { GraphType, LayoutAlgorithm } from "./types";
import { decodeGraph, decodeFocus, type GraphBuffers } from "./graphBuffers";

const BASE = "/api";

// ── бинарный протокол ────────────────────────────────────────────
// ca-AstroPh: 3.1 МБ против 13.0 МБ JSON и ноль JS-объектов на вершину или ребро.

async function postBinary(url: string, body: unknown): Promise<ArrayBuffer> {
  const isRaw = body instanceof ArrayBuffer || ArrayBuffer.isView(body);
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": isRaw ? "application/octet-stream" : "application/json" },
    body: isRaw ? (body as BodyInit) : JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Запрос не выполнен");
  }
  return res.arrayBuffer();
}

export async function generateGraphBin(
  graphType: GraphType, n: number, p: number, m: number, k: number, numEdges?: number,
): Promise<GraphBuffers> {
  return decodeGraph(await postBinary("/graph/generate.bin", {
    graph_type: graphType, n, p, m, k, num_edges: numEdges ?? undefined,
  }));
}

export async function uploadGraphBin(pairs: Uint32Array): Promise<GraphBuffers> {
  return decodeGraph(await postBinary("/graph/upload.bin", pairs));
}

export async function changeLayoutBin(algorithm: LayoutAlgorithm): Promise<GraphBuffers> {
  return decodeGraph(await postBinary("/graph/layout.bin", { algorithm }));
}

export async function collapseGraphBin(minSize: number): Promise<GraphBuffers> {
  return decodeGraph(await postBinary("/graph/collapse.bin", { min_size: minSize }));
}

export async function viewSubgraphBin(communityId: number): Promise<GraphBuffers> {
  return decodeGraph(await postBinary("/graph/subgraph.bin", { community_id: communityId }));
}

export async function focusNodeBin(
  nodeId: number, depth: number,
  preserveBridges: boolean, preserveHubs: boolean, hubThreshold: number,
  target: GraphBuffers,
): Promise<{ alpha: Float32Array; focusNode: number; matched: number }> {
  return decodeFocus(await postBinary("/graph/focus.bin", {
    node_id: nodeId,
    depth,
    preserve_bridges: preserveBridges,
    preserve_hubs: preserveHubs,
    hub_threshold: hubThreshold,
  }), target);
}

/** Сбрасывает серверный режим просмотра сообщества. */
export async function resetSubgraph(): Promise<void> {
  await fetch(`${BASE}/graph/focus/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
}
