import type { GlobalMetrics, NodeData } from "../types";
import { COMMUNITY_COLORS } from "../constants";

interface Props {
  metrics: GlobalMetrics | null;
  topology: string | null;
  recommendedLayout: string | null;
  numCommunities: number | null;
  communitySizes: Record<number, number> | null;
  selectedNode: NodeData | null;
}

const TOPOLOGY_LABELS: Record<string, string> = {
  tree: "Дерево",
  sparse: "Разреженный",
  clustered: "Кластеризованный",
  dense: "Плотный",
  small_world: "Малый мир",
};

const LAYOUT_LABELS: Record<string, string> = {
  spring: "Spring (силовой)",
  circular: "Круговой",
  spectral: "Спектральный",
  kamada_kawai: "Kamada-Kawai",
  shell: "Shell",
  community: "По сообществам",
};

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits);
}

export default function MetricsPanel({
  metrics,
  topology,
  recommendedLayout,
  numCommunities,
  communitySizes,
  selectedNode,
}: Props) {
  if (!metrics) {
    return (
      <aside className="metrics-panel">
        <h2>Метрики</h2>
        <p className="hint">Сгенерируйте граф для просмотра метрик</p>
      </aside>
    );
  }

  return (
    <aside className="metrics-panel">
      <h2>Метрики графа</h2>

      <section>
        <h3>Топология</h3>
        <div className="topology-badge">
          {TOPOLOGY_LABELS[topology ?? ""] ?? topology}
        </div>
        <div className="metric-hint">
          Рекомендован: {LAYOUT_LABELS[recommendedLayout ?? ""] ?? recommendedLayout}
        </div>
      </section>

      <section>
        <h3>Глобальные метрики</h3>
        <table className="metrics-table">
          <tbody>
            <tr>
              <td>Вершины</td>
              <td>{metrics.num_nodes}</td>
            </tr>
            <tr>
              <td>Рёбра</td>
              <td>{metrics.num_edges}</td>
            </tr>
            <tr>
              <td>Плотность</td>
              <td>{fmt(metrics.density)}</td>
            </tr>
            <tr>
              <td>Диаметр</td>
              <td>{metrics.diameter ?? "—"}</td>
            </tr>
            <tr>
              <td>Радиус</td>
              <td>{metrics.radius ?? "—"}</td>
            </tr>
            <tr>
              <td>Коэф. кластеризации</td>
              <td>{fmt(metrics.clustering_coefficient)}</td>
            </tr>
            <tr>
              <td>Модулярность</td>
              <td>{fmt(metrics.modularity)}</td>
            </tr>
            <tr>
              <td>Ср. степень</td>
              <td>{fmt(metrics.avg_degree, 2)}</td>
            </tr>
            <tr>
              <td>Ср. длина пути</td>
              <td>{fmt(metrics.avg_path_length)}</td>
            </tr>
            <tr>
              <td>Компоненты</td>
              <td>{metrics.num_components}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h3>Сообщества ({numCommunities ?? 0})</h3>
        {communitySizes && (
          <div className="community-list">
            {Object.entries(communitySizes)
              .sort(([, a], [, b]) => b - a)
              .map(([id, size]) => (
                <div key={id} className="community-item">
                  <span
                    className="community-dot"
                    style={{
                      backgroundColor: COMMUNITY_COLORS[+id % COMMUNITY_COLORS.length],
                    }}
                  />
                  <span>#{id}</span>
                  <span className="community-size">{size} узл.</span>
                </div>
              ))}
          </div>
        )}
      </section>

      {selectedNode && (
        <section>
          <h3>Узел {selectedNode.label}</h3>
          <table className="metrics-table">
            <tbody>
              <tr>
                <td>Степень</td>
                <td>{selectedNode.degree}</td>
              </tr>
              <tr>
                <td>Центр. степени</td>
                <td>{fmt(selectedNode.degree_centrality)}</td>
              </tr>
              <tr>
                <td>Центр. посредн.</td>
                <td>{fmt(selectedNode.betweenness_centrality)}</td>
              </tr>
              <tr>
                <td>Центр. близости</td>
                <td>{fmt(selectedNode.closeness_centrality)}</td>
              </tr>
              <tr>
                <td>Собств. центр.</td>
                <td>{fmt(selectedNode.eigenvector_centrality)}</td>
              </tr>
              <tr>
                <td>Сообщество</td>
                <td>#{selectedNode.community}</td>
              </tr>
            </tbody>
          </table>
        </section>
      )}
    </aside>
  );
}

