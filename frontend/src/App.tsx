import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import GraphGL from "./components/GraphGL";
import ControlPanel from "./components/ControlPanel";
import MetricsPanel from "./components/MetricsPanel";
import PhysicsPanel from "./components/PhysicsPanel";
import * as api from "./api";
import { useLiveLayout, DEFAULT_PARAMS, type SimParams } from "./useLiveLayout";
import { degreeAsFloat, shuffleEdges, type GraphBuffers } from "./graphBuffers";
import type {
  CentralityMetric,
  GraphType,
  LayoutAlgorithm,
  NodeData,
  GlobalMetrics,
} from "./types";
import "./App.css";

/** Граф целиком живёт в типизированных массивах, а не в объектах React. */
export default function App() {
  const [graph, setGraph] = useState<GraphBuffers | null>(null);
  const [fullGraph, setFullGraph] = useState<GraphBuffers | null>(null);
  const [parentGraph, setParentGraph] = useState<GraphBuffers | null>(null);
  const [focusAlpha, setFocusAlpha] = useState<Float32Array | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sizeMetric, setSizeMetric] = useState<CentralityMetric>("degree_centrality");
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [currentLayout, setCurrentLayout] = useState("");
  const [preserveBridges, setPreserveBridges] = useState(true);
  const [preserveHubs, setPreserveHubs] = useState(true);
  const [hubThreshold, setHubThreshold] = useState(0.8);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [viewingCommunity, setViewingCommunity] = useState<number | null>(null);

  // физика
  const [physicsOn, setPhysicsOn] = useState(true);
  const [params, setParams] = useState<SimParams>(DEFAULT_PARAMS);
  // Настройки рёбер убраны из интерфейса: яркость подбирает рендерер, тон дублировал цвет вершин.
  const EDGE_OPACITY = 1;
  const EDGE_TINT = 0;

  const focusDepthRef = useRef(2);
  // Параметры фокуса продублированы в ref: запрос уходит до перерисовки, и из
  // замыкания читалось предыдущее значение — галочки работали ровно наоборот.
  const focusParamsRef = useRef({ bridges: true, hubs: true, threshold: 0.8 });

  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(null), 6000);
    return () => clearTimeout(timer);
  }, [error]);

  const liveGraph = useMemo(() => {
    if (!graph) return null;
    const src = new Uint32Array(graph.m);
    const tgt = new Uint32Array(graph.m);
    for (let e = 0; e < graph.m; e++) {
      src[e] = graph.edges[e * 2];
      tgt[e] = graph.edges[e * 2 + 1];
    }
    return {
      n: graph.n,
      pos: graph.xy,
      src,
      tgt,
      weight: graph.weights.length === graph.m ? graph.weights : null,
      degree: degreeAsFloat(graph),
    };
  }, [graph]);

  const live = useLiveLayout(liveGraph, physicsOn && !!graph);
  const { setParams: pushParams, reheat, onFrame, drag, release, statsRef } = live;

  const liveHandle = useMemo(
    () => ({ onFrame, drag, release }),
    [onFrame, drag, release],
  );

  const handleParamsChange = useCallback(
    (patch: Partial<SimParams>) => {
      setParams((p) => ({ ...p, ...patch }));
      pushParams(patch);
    },
    [pushParams],
  );

  const clearFocus = useCallback(() => {
    setFocusAlpha(null);
    setSelectedNodeId(null);
  }, []);

  const adopt = useCallback(
    (g: GraphBuffers, opts: { full?: boolean; collapsed?: boolean } = {}) => {
      // перемешиваем рёбра один раз: любой префикс индексного буфера — честная выборка для LOD
      shuffleEdges(g);
      setGraph(g);
      if (opts.full !== false) setFullGraph(g);
      setIsCollapsed(!!opts.collapsed);
      setCurrentLayout(g.meta.recommendedLayout);
    },
    [],
  );

  const run = useCallback(
    async (fn: () => Promise<void>, what: string) => {
      setLoading(true);
      setError(null);
      try {
        await fn();
      } catch (e) {
        setError(e instanceof Error ? e.message : what);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleGenerate = useCallback(
    (type: GraphType, n: number, p: number, m: number, k: number, numEdges?: number) =>
      run(async () => {
        clearFocus();
        setParentGraph(null);
        setViewingCommunity(null);
        adopt(await api.generateGraphBin(type, n, p, m, k, numEdges));
      }, "Ошибка генерации"),
    [run, clearFocus, adopt],
  );

  const handleUpload = useCallback(
    (pairs: Uint32Array) =>
      run(async () => {
        clearFocus();
        setParentGraph(null);
        setViewingCommunity(null);
        adopt(await api.uploadGraphBin(pairs));
      }, "Ошибка загрузки графа"),
    [run, clearFocus, adopt],
  );

  const handleLayoutChange = useCallback(
    (algorithm: LayoutAlgorithm) =>
      run(async () => {
        clearFocus();
        setParentGraph(null);
        setViewingCommunity(null);
        const g = await api.changeLayoutBin(algorithm);
        shuffleEdges(g);
        setGraph(g);
        setFullGraph(g);
        setIsCollapsed(false);
        setCurrentLayout(algorithm);
      }, "Ошибка смены укладки"),
    [run, clearFocus],
  );

  const handleCollapse = useCallback(
    (minSize: number) =>
      run(async () => {
        clearFocus();
        setParentGraph(null);
        setViewingCommunity(null);
        const g = await api.collapseGraphBin(minSize);
        shuffleEdges(g);
        setGraph(g);
        setIsCollapsed(true);
      }, "Ошибка свёртки"),
    [run, clearFocus],
  );

  const handleExpandAll = useCallback(() => {
    if (!fullGraph) return;
    clearFocus();
    setGraph(fullGraph);
    setIsCollapsed(false);
    setParentGraph(null);
    setViewingCommunity(null);
  }, [fullGraph, clearFocus]);

  const requestFocus = useCallback(
    async (nodeId: number, target: GraphBuffers) => {
      try {
        const p = focusParamsRef.current;
        const { alpha } = await api.focusNodeBin(
          nodeId,
          focusDepthRef.current,
          p.bridges,
          p.hubs,
          p.threshold,
          target,
        );
        setFocusAlpha(alpha);
      } catch {
        /* фокус не критичен: молча остаёмся без затенения */
      }
    },
    [],
  );

  const handleNodeClick = useCallback(
    (nodeId: number) => {
      if (!graph) return;
      const idx = graph.indexOf.get(nodeId);
      if (idx !== undefined && graph.isCollapsed[idx]) {
        const cid = graph.community[idx];
        run(async () => {
          clearFocus();
          setParentGraph(graph);
          setViewingCommunity(cid);
          const g = await api.viewSubgraphBin(cid);
          shuffleEdges(g);
          setGraph(g);
          setIsCollapsed(false);
        }, "Не удалось открыть сообщество");
        return;
      }
      if (selectedNodeId === nodeId) {
        clearFocus();
        return;
      }
      setSelectedNodeId(nodeId);
      requestFocus(nodeId, graph);
    },
    [graph, selectedNodeId, clearFocus, requestFocus, run],
  );

  const handleGoBack = useCallback(() => {
    if (!parentGraph) return;
    clearFocus();
    // сервер держит режим просмотра сообщества у себя, выход подтверждаем явно
    api.resetSubgraph().catch(() => {});
    setGraph(parentGraph);
    setParentGraph(null);
    setViewingCommunity(null);
    setIsCollapsed(true);
  }, [parentGraph, clearFocus]);

  const handleFocusDepthChange = useCallback(
    (depth: number) => {
      focusDepthRef.current = depth;
      if (selectedNodeId !== null && graph) requestFocus(selectedNodeId, graph);
    },
    [selectedNodeId, requestFocus],
  );

  const refocus = useCallback(
    (apply: () => void) => {
      apply();
      if (selectedNodeId !== null && graph) requestFocus(selectedNodeId, graph);
    },
    [selectedNodeId, graph, requestFocus],
  );

  // ── данные для боковых панелей ────────────────────────────────
  const selectedNode: NodeData | null = useMemo(() => {
    if (!graph || selectedNodeId === null) return null;
    const i = graph.indexOf.get(selectedNodeId);
    if (i === undefined) return null;
    return {
      id: graph.ids[i],
      label: String(graph.ids[i]),
      x: graph.xy[i * 2],
      y: graph.xy[i * 2 + 1],
      degree_centrality: graph.degreeCentrality[i],
      betweenness_centrality: graph.betweennessCentrality[i],
      closeness_centrality: graph.closenessCentrality[i],
      eigenvector_centrality: graph.eigenvectorCentrality[i],
      community: graph.community[i],
      degree: graph.degree[i],
      is_collapsed: !!graph.isCollapsed[i],
      member_count: graph.memberCount[i],
    };
  }, [graph, selectedNodeId]);

  const metrics = (graph?.meta.globalMetrics ?? null) as GlobalMetrics | null;

  return (
    <div className="app">
      <ControlPanel
        onGenerate={handleGenerate}
        onLayoutChange={handleLayoutChange}
        onSizeMetricChange={setSizeMetric}
        onFocusDepthChange={handleFocusDepthChange}
        onResetFocus={clearFocus}
        onUpload={handleUpload}
        onPreserveBridgesChange={(v) => refocus(() => {
          focusParamsRef.current.bridges = v;
          setPreserveBridges(v);
        })}
        onPreserveHubsChange={(v) => refocus(() => {
          focusParamsRef.current.hubs = v;
          setPreserveHubs(v);
        })}
        onHubThresholdChange={(v) => refocus(() => {
          focusParamsRef.current.threshold = v;
          setHubThreshold(v);
        })}
        onCollapse={handleCollapse}
        onExpandAll={handleExpandAll}
        onGoBack={handleGoBack}
        currentLayout={currentLayout}
        loading={loading}
        focusActive={focusAlpha !== null}
        preserveBridges={preserveBridges}
        preserveHubs={preserveHubs}
        hubThreshold={hubThreshold}
        isCollapsed={isCollapsed}
        hasGraph={!!graph}
        viewingCommunity={viewingCommunity}
        physics={
          graph ? (
            <PhysicsPanel
              enabled={physicsOn}
              onEnabledChange={setPhysicsOn}
              params={params}
              onParamsChange={handleParamsChange}
              onReheat={() => reheat(1)}
              statsRef={statsRef}
            />
          ) : null
        }
      />

      <main className="main-area">
        {error && <div className="error-toast">{error}</div>}
        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
            <p>Обработка графа...</p>
          </div>
        )}

        {!graph && !loading && (
          <div className="placeholder">
            <h1>Smart Focus</h1>
            <p>Адаптивная визуализация графов</p>
            <p className="hint">Сгенерируйте или загрузите граф</p>
          </div>
        )}

        {viewingCommunity !== null && (
          <div className="subgraph-banner">
            <span>
              Сообщество #{viewingCommunity} — {graph?.n} вершин
            </span>
            <button onClick={handleGoBack} disabled={loading}>
              ← Назад
            </button>
          </div>
        )}

        {graph && (
          <GraphGL
            key={viewingCommunity !== null ? `sub-${viewingCommunity}` : "main"}
            graph={graph}
            focusAlpha={focusAlpha}
            sizeMetric={sizeMetric}
            selectedNode={selectedNodeId}
            onNodeClick={handleNodeClick}
            live={physicsOn ? liveHandle : null}
            edgeTint={EDGE_TINT}
            edgeOpacity={EDGE_OPACITY}
          />
        )}
      </main>

      <MetricsPanel
        metrics={metrics}
        topology={graph?.meta.topology ?? null}
        recommendedLayout={graph?.meta.recommendedLayout ?? null}
        numCommunities={graph?.meta.numCommunities ?? null}
        communitySizes={graph?.meta.communitySizes ?? null}
        selectedNode={selectedNode}
      />
    </div>
  );
}
