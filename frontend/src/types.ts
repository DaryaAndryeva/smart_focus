export interface NodeData {
  id: number;
  label: string;
  x: number;
  y: number;
  degree_centrality: number;
  betweenness_centrality: number;
  closeness_centrality: number;
  eigenvector_centrality: number;
  community: number;
  degree: number;
  is_collapsed: boolean;
  member_count: number;
}

export interface EdgeData {
  source: number;
  target: number;
  weight: number;
}

export interface GlobalMetrics {
  num_nodes: number;
  num_edges: number;
  density: number;
  diameter: number | null;
  radius: number | null;
  clustering_coefficient: number;
  num_components: number;
  modularity: number;
  avg_degree: number;
  avg_path_length: number | null;
}

export interface GraphData {
  nodes: NodeData[];
  edges: EdgeData[];
  global_metrics: GlobalMetrics;
  topology: string;
  recommended_layout: string;
  num_communities: number;
  community_sizes: Record<number, number>;
}

export interface FocusResult {
  node_opacities: Record<number, number>;
  edge_opacities: Record<string, number>;
  context_nodes: number[];
  focus_node: number;
}

export type CentralityMetric =
  | "degree_centrality"
  | "betweenness_centrality"
  | "closeness_centrality"
  | "eigenvector_centrality";

export type GraphType =
  | "dense_gnm"
  | "random"
  | "scale_free"
  | "small_world"
  | "powerlaw_cluster"
  | "random_partition"
  | "stochastic_block"
  | "balanced_tree"
  | "internet_as"
  | "grid";

export type LayoutAlgorithm =
  | "spring"
  | "circular"
  | "spectral"
  | "kamada_kawai"
  | "shell"
  | "community"
  | "multilevel";
