import NumberField from "./NumberField";
import { useState, useRef } from "react";
import type { GraphType, LayoutAlgorithm, CentralityMetric } from "../types";

interface Props {
  onGenerate: (type: GraphType, n: number, p: number, m: number, k: number, numEdges?: number) => void;
  onLayoutChange: (algorithm: LayoutAlgorithm) => void;
  onSizeMetricChange: (metric: CentralityMetric) => void;
  onFocusDepthChange: (depth: number) => void;
  onResetFocus: () => void;
  onUpload: (pairs: Uint32Array) => void;
  physics?: React.ReactNode;
  onPreserveBridgesChange: (v: boolean) => void;
  onPreserveHubsChange: (v: boolean) => void;
  onHubThresholdChange: (v: number) => void;
  onCollapse: (minSize: number) => void;
  onExpandAll: () => void;
  onGoBack: () => void;
  currentLayout: string;
  loading: boolean;
  focusActive: boolean;
  preserveBridges: boolean;
  preserveHubs: boolean;
  hubThreshold: number;
  isCollapsed: boolean;
  hasGraph: boolean;
  viewingCommunity: number | null;
}

const GRAPH_TYPES: { value: GraphType; label: string }[] = [
  { value: "dense_gnm", label: "Плотный G(n,m)" },
  { value: "random", label: "Случайный (Эрдёш-Реньи)" },
  { value: "scale_free", label: "Безмасштабный (Барабаши-Альберт)" },
  { value: "small_world", label: "Малый мир (Уоттс-Строгац)" },
  { value: "powerlaw_cluster", label: "Степенной кластерный (Холме-Ким)" },
  { value: "random_partition", label: "Случайное разбиение" },
  { value: "stochastic_block", label: "Стохастическая блочная модель" },
  { value: "balanced_tree", label: "Сбалансированное дерево" },
  { value: "internet_as", label: "Интернет AS-граф" },
  { value: "grid", label: "Решётка" },
];

const LAYOUTS: { value: LayoutAlgorithm; label: string }[] = [
  { value: "spring", label: "Spring" },
  { value: "circular", label: "Круговой" },
  { value: "spectral", label: "Спектральный" },
  { value: "kamada_kawai", label: "Kamada-Kawai" },
  { value: "shell", label: "Shell" },
  { value: "community", label: "По сообществам" },
  { value: "multilevel", label: "Многоуровневый" },
];

const SIZE_METRICS: { value: CentralityMetric; label: string }[] = [
  { value: "degree_centrality", label: "Degree" },
  { value: "betweenness_centrality", label: "Betweenness" },
  { value: "closeness_centrality", label: "Closeness" },
  { value: "eigenvector_centrality", label: "Eigenvector" },
];

/** Разбирает список рёбер сразу в Uint32Array: посимвольно, без split —
 *  массив строк для файла в 5 МБ занимал 15 МБ временной памяти. */
function parseEdgeList(text: string): { pairs: Uint32Array; nodes: number } {
  const len = text.length;
  let cap = 1 << 16;
  let arr = new Uint32Array(cap);
  let count = 0;
  const seen = new Set<number>();
  let i = 0;

  while (i < len) {
    const c = text.charCodeAt(i);
    // строки комментариев пропускаем целиком
    if (c === 35 || (c === 47 && text.charCodeAt(i + 1) === 47)) {
      while (i < len && text.charCodeAt(i) !== 10) i++;
      i++;
      continue;
    }
    let nums = 0;
    let a = 0;
    let b = 0;
    while (i < len) {
      const ch = text.charCodeAt(i);
      if (ch === 10) { i++; break; }
      if (ch >= 48 && ch <= 57) {
        let v = 0;
        while (i < len) {
          const d = text.charCodeAt(i);
          if (d < 48 || d > 57) break;
          v = v * 10 + (d - 48);
          i++;
        }
        if (nums === 0) a = v;
        else if (nums === 1) b = v;
        nums++;
      } else {
        i++;
      }
    }
    if (nums >= 2) {
      if (count + 2 > cap) {
        cap *= 2;
        const next = new Uint32Array(cap);
        next.set(arr);
        arr = next;
      }
      arr[count++] = a;
      arr[count++] = b;
      seen.add(a);
      seen.add(b);
    }
  }
  return { pairs: arr.subarray(0, count), nodes: seen.size };
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} Б`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} КБ`;
  return `${(n / 1024 / 1024).toFixed(1)} МБ`;
}

export default function ControlPanel({
  onGenerate,
  onLayoutChange,
  onSizeMetricChange,
  onFocusDepthChange,
  onResetFocus,
  onUpload,
  onPreserveBridgesChange,
  onPreserveHubsChange,
  onHubThresholdChange,
  onCollapse,
  onExpandAll,
  onGoBack,
  currentLayout,
  loading,
  focusActive,
  preserveBridges,
  preserveHubs,
  hubThreshold,
  isCollapsed,
  hasGraph,
  viewingCommunity,
  physics,
}: Props) {
  const [graphType, setGraphType] = useState<GraphType>("dense_gnm");
  const [nodeCount, setNodeCount] = useState(50);
  const [paramP, setParamP] = useState(0.1);
  const [paramM, setParamM] = useState(3);
  const [paramK, setParamK] = useState(4);
  const [numEdges, setNumEdges] = useState(100);
  const [sizeMetric, setSizeMetric] = useState<CentralityMetric>("degree_centrality");
  const [focusDepth, setFocusDepth] = useState(2);

  const [collapseThreshold, setCollapseThreshold] = useState(10);

  const [genOpen, setGenOpen] = useState(false);
  const [edgeText, setEdgeText] = useState("");
  const [status, setStatus] = useState<
    { text: string; kind: "info" | "ok" | "error" } | null
  >(null);
  const [fileInfo, setFileInfo] = useState<
    { name: string; size: number; nodes?: number; edges?: number } | null
  >(null);
  const [readPercent, setReadPercent] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const parsedRef = useRef<{ pairs: Uint32Array; nodes: number } | null>(null);
  const pendingUploadRef = useRef(false);
  const [reading, setReading] = useState(false);

  const maxEdges = Math.floor(nodeCount * (nodeCount - 1) / 2);

  const handleGenerate = () => {
    onGenerate(
      graphType, nodeCount, paramP, paramM, paramK,
      graphType === "dense_gnm" ? numEdges : undefined,
    );
  };

  const handleUpload = () => {
    if (reading) {
      // намерение запоминаем и отправляем сами по окончании разбора
      pendingUploadRef.current = true;
      setStatus({ text: "Файл ещё читается — загрузка начнётся сама", kind: "info" });
      return;
    }

    // только при пустом поле: иначе набранный вручную список молча подменится файлом
    const ready = parsedRef.current;
    if (ready && ready.pairs.length && !edgeText.trim()) {
      setStatus({ text: `Отправляю ${ready.nodes} вершин и ${ready.pairs.length / 2} рёбер…`, kind: "info" });
      onUpload(ready.pairs);
      return;
    }

    if (!edgeText.trim()) {
      setStatus({ text: "Выберите файл или вставьте список рёбер", kind: "error" });
      return;
    }
    const { pairs, nodes } = parseEdgeList(edgeText);
    if (pairs.length === 0) {
      setStatus({ text: "В тексте не распознано ни одного ребра", kind: "error" });
      return;
    }
    setStatus({ text: `Отправляю ${nodes} вершин и ${pairs.length / 2} рёбер…`, kind: "info" });
    onUpload(pairs);
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // сброс значения: иначе повторный выбор того же файла не вызовет change
    e.target.value = "";
    if (!file) return;

    // сбрасываем сразу: иначе клик во время чтения отправит предыдущий файл
    parsedRef.current = null;
    pendingUploadRef.current = false;
    setEdgeText("");
    setReading(true);
    setReadPercent(0);
    setFileInfo({ name: file.name, size: file.size });
    setStatus({ text: "Читаю файл…", kind: "info" });

    const reader = new FileReader();
    reader.onprogress = (ev) => {
      if (ev.lengthComputable) setReadPercent(Math.round((ev.loaded / ev.total) * 100));
    };
    reader.onerror = () => {
      setReading(false);
      setReadPercent(null);
      setStatus({ text: "Не удалось прочитать файл", kind: "error" });
    };
    reader.onload = (ev) => {
      const text = (ev.target?.result as string) ?? "";
      const t0 = performance.now();
      const parsed = parseEdgeList(text);
      setReading(false);
      setReadPercent(null);
      if (parsed.pairs.length === 0) {
        parsedRef.current = null;
        setStatus({ text: "В файле не найдено ни одного ребра", kind: "error" });
        return;
      }
      parsedRef.current = parsed;
      const edges = parsed.pairs.length / 2;
      setFileInfo({ name: file.name, size: file.size, nodes: parsed.nodes, edges });
      setStatus({
        text: `Разобран за ${Math.round(performance.now() - t0)} мс. Нажмите «Загрузить»`,
        kind: "ok",
      });
      if (pendingUploadRef.current) {
        pendingUploadRef.current = false;
        setStatus({ text: `Отправляю ${parsed.nodes} вершин и ${edges} рёбер…`, kind: "info" });
        onUpload(parsed.pairs);
      }
    };
    reader.readAsText(file);
  };

  return (
    <aside className="control-panel">
      <h2>Smart Focus</h2>

      {viewingCommunity !== null && (
        <section className="subgraph-nav">
          <p className="hint" style={{ marginBottom: 6 }}>
            Просмотр сообщества #{viewingCommunity}
          </p>
          <button className="btn-secondary" onClick={onGoBack} disabled={loading}>
            ← Назад к общему графу
          </button>
        </section>
      )}

      {/* ── Generate (collapsible) ── */}
      <section className="collapsible-section">
        <button className="section-toggle" onClick={() => setGenOpen(!genOpen)}>
          <h3>Генерация графа</h3>
          <span className={`chevron ${genOpen ? "open" : ""}`}>›</span>
        </button>

        {genOpen && (
          <div className="section-body">
            <label>
              Тип графа
              <select value={graphType} onChange={(e) => setGraphType(e.target.value as GraphType)}>
                {GRAPH_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </label>

            <NumberField
              label="Вершин"
              value={nodeCount}
              min={5}
              max={100000}
              step={100}
              onChange={(v) => {
                setNodeCount(v);
                const mx = Math.floor(v * (v - 1) / 2);
                if (numEdges > mx) setNumEdges(mx);
              }}
            />

            {graphType === "dense_gnm" && (
              <NumberField
                label="Рёбер"
                value={numEdges}
                min={1}
                max={maxEdges}
                step={100}
                onChange={setNumEdges}
              />
            )}

            {["random", "small_world", "powerlaw_cluster", "random_partition", "stochastic_block"].includes(graphType) && (
              <NumberField label="Вероятность p" value={paramP}
                min={0.01} max={0.5} step={0.01} onChange={setParamP} />
            )}

            {["scale_free", "powerlaw_cluster", "balanced_tree"].includes(graphType) && (
              <NumberField
                label={graphType === "balanced_tree" ? "Ветвление (r)" : "Рёбер на узел (m)"}
                value={paramM} min={1} max={10} step={1} onChange={setParamM} />
            )}

            {["small_world", "random_partition", "stochastic_block"].includes(graphType) && (
              <NumberField
                label={graphType === "small_world" ? "Соседей (k)" : "Групп (k)"}
                value={paramK} min={2} max={10} step={1} onChange={setParamK} />
            )}

            <button className="btn-primary" onClick={handleGenerate} disabled={loading}>
              {loading ? "Загрузка..." : "Сгенерировать"}
            </button>
          </div>
        )}
      </section>

      {/* ── Upload ── */}
      <section>
        <h3>Загрузка графа</h3>
        <textarea
          className="edge-textarea"
          placeholder={"Список рёбер (одно на строку):\n0 1\n1 2\n2 3\n0,3"}
          value={edgeText}
          onChange={(e) => {
            setEdgeText(e.target.value);
            // набранный вручную список имеет приоритет над выбранным файлом
            if (e.target.value.trim()) parsedRef.current = null;
          }}
          rows={5}
        />
        <div className="upload-actions">
          <label className="btn-file">
            {fileInfo ? "Другой файл" : "Выбрать файл"}
            <input ref={fileRef} type="file" accept=".txt,.csv,.tsv,.edgelist,.edges"
              onChange={handleFile} hidden />
          </label>
          <button className="btn-primary" style={{ flex: 1 }} onClick={handleUpload} disabled={loading}>
            {loading ? "Загрузка..." : reading ? "Чтение файла..." : "Загрузить"}
          </button>
        </div>

        {fileInfo && (
          <div className={`file-card${edgeText.trim() ? " file-card-muted" : ""}`}>
            <div className="file-card-row">
              <span className="file-card-name" title={fileInfo.name}>{fileInfo.name}</span>
              <button
                className="file-card-clear"
                title="Убрать файл"
                onClick={() => {
                  parsedRef.current = null;
                  pendingUploadRef.current = false;
                  setFileInfo(null);
                  setStatus(null);
                }}
              >×</button>
            </div>
            <div className="file-card-meta">
              {formatBytes(fileInfo.size)}
              {fileInfo.nodes !== undefined && (
                <> · {fileInfo.nodes.toLocaleString("ru")} вершин · {fileInfo.edges!.toLocaleString("ru")} рёбер</>
              )}
            </div>
            {reading && (
              <div className="file-progress">
                <div className="file-progress-bar" style={{ width: `${readPercent ?? 0}%` }} />
              </div>
            )}
            {edgeText.trim() && (
              <div className="file-card-note">
                Поле ниже не пустое — будет загружен текст из него, а не файл
              </div>
            )}
          </div>
        )}

        {status && <div className={`parse-info parse-info-${status.kind}`}>{status.text}</div>}
      </section>

      {physics}

      {/* ── Layout ── */}
      <section>
        <h3>Алгоритм укладки</h3>
        <div className="layout-grid">
          {LAYOUTS.map((l) => (
            <button key={l.value}
              className={`btn-layout ${currentLayout === l.value ? "active" : ""}`}
              onClick={() => onLayoutChange(l.value)} disabled={loading}>
              {l.label}
            </button>
          ))}
        </div>
      </section>

      {/* ── Collapse ── */}
      {hasGraph && (
        <section>
          <h3>Свёртка сообществ</h3>
          <p className="hint">Объединить крупные сообщества в мета-узлы</p>
          <NumberField label="Порог" value={collapseThreshold} suffix="узлов"
            min={2} max={5000} step={10} onChange={setCollapseThreshold} />
          <div className="layout-grid">
            <button className="btn-primary" style={{fontSize: 12}} onClick={() => onCollapse(collapseThreshold)} disabled={loading}>
              Свернуть
            </button>
            {isCollapsed && (
              <button className="btn-secondary" style={{fontSize: 12, marginTop: 0}} onClick={onExpandAll} disabled={loading}>
                Развернуть
              </button>
            )}
          </div>
        </section>
      )}

      {/* ── Size metric ── */}
      <section>
        <h3>Размер узлов</h3>
        <div className="layout-grid">
          {SIZE_METRICS.map((m) => (
            <button key={m.value}
              className={`btn-layout ${sizeMetric === m.value ? "active" : ""}`}
              onClick={() => { setSizeMetric(m.value); onSizeMetricChange(m.value); }}>
              {m.label}
            </button>
          ))}
        </div>
      </section>

      {/* ── Focus ── */}
      <section>
        <h3>Управление фокусом</h3>
        <p className="hint">Кликните на узел для фокусировки</p>

        <NumberField label="Глубина" value={focusDepth} min={1} max={5} step={1}
          onChange={(v) => { setFocusDepth(v); onFocusDepthChange(v); }} />

        <label className="checkbox-label">
          <input type="checkbox" checked={preserveBridges}
            onChange={(e) => onPreserveBridgesChange(e.target.checked)} />
          Сохранять мосты
        </label>

        <label className="checkbox-label">
          <input type="checkbox" checked={preserveHubs}
            onChange={(e) => onPreserveHubsChange(e.target.checked)} />
          Сохранять хабы
        </label>

        {preserveHubs && (
          <NumberField label="Порог хабов" value={hubThreshold}
            min={0.5} max={0.99} step={0.01} onChange={onHubThresholdChange} />
        )}

        {focusActive && (
          <button className="btn-secondary" onClick={onResetFocus}>Сбросить фокус</button>
        )}
      </section>
    </aside>
  );
}
