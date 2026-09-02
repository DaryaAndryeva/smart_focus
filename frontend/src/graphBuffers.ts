/**
 * Граф в плоских типизированных массивах: те же ArrayBuffer служат и вершинными буферами
 * WebGL, и рабочими массивами силовой симуляции в воркере. Массив объектов NodeData/EdgeData
 * для ca-AstroPh давал 216 822 объекта JavaScript и ~62 МБ кучи при 1.9 МБ полезных данных.
 */

import type { CentralityMetric } from "./types";

export interface GraphBuffers {
  n: number;
  m: number;
  ids: Int32Array;
  /** координаты, чередующиеся x, y — Float32Array(2n) */
  xy: Float32Array;
  community: Uint32Array;
  degree: Uint32Array;
  degreeCentrality: Float32Array;
  betweennessCentrality: Float32Array;
  closenessCentrality: Float32Array;
  eigenvectorCentrality: Float32Array;
  /** число вершин внутри мета-узла (1 для обычных) */
  memberCount: Uint32Array;
  isCollapsed: Uint8Array;
  /** рёбра как ИНДЕКСЫ вершин, чередующиеся s, t — Uint32Array(2m) */
  edges: Uint32Array;
  weights: Float32Array;
  /** идентификатор вершины -> её индекс */
  indexOf: Map<number, number>;
  meta: GraphMeta;
}

export interface GraphMeta {
  topology: string;
  recommendedLayout: string;
  numCommunities: number;
  communitySizes: Record<number, number>;
  globalMetrics: Record<string, number | null>;
}

const MAGIC = 0x31474653; // "SFG1" little-endian

interface ArraySpec {
  name: string;
  dtype: string;
  count: number;
  offset: number;
}

function viewOf(buf: ArrayBuffer, base: number, spec: ArraySpec) {
  const at = base + spec.offset;
  switch (spec.dtype) {
    case "<i4":
      return new Int32Array(buf, at, spec.count);
    case "<u4":
      return new Uint32Array(buf, at, spec.count);
    case "<f4":
      return new Float32Array(buf, at, spec.count);
    case "<u1":
      return new Uint8Array(buf, at, spec.count);
    default:
      throw new Error(`неизвестный тип массива: ${spec.dtype}`);
  }
}

/** Разбирает бинарный ответ сервера без единой аллокации объекта на вершину. */
export function decodeGraph(buf: ArrayBuffer): GraphBuffers {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("неверная сигнатура бинарного графа");
  const hdrLen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hdrLen)));
  const base = 8 + hdrLen;

  const byName = new Map<string, ArraySpec>();
  for (const spec of header.arrays as ArraySpec[]) byName.set(spec.name, spec);
  const get = (name: string) => {
    const spec = byName.get(name);
    if (!spec) throw new Error(`в заголовке нет массива ${name}`);
    return viewOf(buf, base, spec);
  };

  const n: number = header.n;
  const m: number = header.m;

  const ids = get("id") as Int32Array;
  const x = get("x") as Float32Array;
  const y = get("y") as Float32Array;

  const xy = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    xy[i * 2] = x[i];
    xy[i * 2 + 1] = y[i];
  }

  const src = get("source") as Uint32Array;
  const tgt = get("target") as Uint32Array;
  const edges = new Uint32Array(m * 2);
  for (let e = 0; e < m; e++) {
    edges[e * 2] = src[e];
    edges[e * 2 + 1] = tgt[e];
  }

  const indexOf = new Map<number, number>();
  for (let i = 0; i < n; i++) indexOf.set(ids[i], i);

  const sizes: Record<number, number> = {};
  for (const [k, v] of Object.entries(header.community_sizes ?? {})) {
    sizes[Number(k)] = v as number;
  }

  return {
    n,
    m,
    ids,
    xy,
    community: get("community") as Uint32Array,
    degree: get("degree") as Uint32Array,
    degreeCentrality: get("degree_centrality") as Float32Array,
    betweennessCentrality: get("betweenness_centrality") as Float32Array,
    closenessCentrality: get("closeness_centrality") as Float32Array,
    eigenvectorCentrality: get("eigenvector_centrality") as Float32Array,
    memberCount: get("member_count") as Uint32Array,
    isCollapsed: get("is_collapsed") as Uint8Array,
    edges,
    weights: get("weight") as Float32Array,
    indexOf,
    meta: {
      topology: header.topology,
      recommendedLayout: header.recommended_layout,
      numCommunities: header.num_communities,
      communitySizes: sizes,
      globalMetrics: header.global_metrics,
    },
  };
}

/** Разбирает бинарный ответ фокуса: прозрачность на каждую вершину. */
export function decodeFocus(
  buf: ArrayBuffer,
  target: GraphBuffers,
): { alpha: Float32Array; focusNode: number; matched: number } {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("неверная сигнатура фокуса");
  const hdrLen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hdrLen)));
  const base = 8 + hdrLen;
  const byName = new Map<string, ArraySpec>();
  for (const spec of header.arrays as ArraySpec[]) byName.set(spec.name, spec);

  const alphaSpec = byName.get("opacity")!;
  const raw = new Float32Array(buf, base + alphaSpec.offset, alphaSpec.count);

  // На экране может быть подмножество серверного графа (свёртка, отдельное сообщество), поэтому
  // раскладка идёт по идентификаторам, а не по позиции; мета-узлы свёртки остаются видимыми.
  const idSpec = byName.get("id");
  const out = new Float32Array(target.n).fill(1);
  let matched = 0;
  if (idSpec) {
    const ids = new Int32Array(buf, base + idSpec.offset, idSpec.count);
    for (let i = 0; i < ids.length; i++) {
      const idx = target.indexOf.get(ids[i]);
      if (idx !== undefined) {
        out[idx] = raw[i];
        matched++;
      }
    }
  }
  return { alpha: out, focusNode: header.focus_node, matched };
}

const METRIC_FIELD: Record<CentralityMetric, keyof GraphBuffers> = {
  degree_centrality: "degreeCentrality",
  betweenness_centrality: "betweennessCentrality",
  closeness_centrality: "closenessCentrality",
  eigenvector_centrality: "eigenvectorCentrality",
};

export function metricArray(g: GraphBuffers, metric: CentralityMetric): Float32Array {
  return g[METRIC_FIELD[metric]] as Float32Array;
}

/** Степени вершин как Float32Array — воркер использует их как массы. */
export function degreeAsFloat(g: GraphBuffers): Float32Array {
  const out = new Float32Array(g.n);
  for (let i = 0; i < g.n; i++) out[i] = g.degree[i];
  return out;
}

/** Перемешивает рёбра один раз, чтобы любой префикс индексного буфера был
 *  равномерной случайной выборкой: на этом держится LOD при отдалении. */
export function shuffleEdges(g: GraphBuffers, seed = 12345): void {
  let s = seed >>> 0;
  const rnd = () => {
    // xorshift32
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  const e = g.edges;
  const w = g.weights;
  for (let i = g.m - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const a0 = e[i * 2], a1 = e[i * 2 + 1];
    e[i * 2] = e[j * 2]; e[i * 2 + 1] = e[j * 2 + 1];
    e[j * 2] = a0; e[j * 2 + 1] = a1;
    if (w.length === g.m) {
      const tw = w[i]; w[i] = w[j]; w[j] = tw;
    }
  }
}
