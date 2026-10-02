"""Многоуровневый силовой алгоритм укладки для графов 10^4-10^5 вершин.

1. Многоуровневая схема (Walshaw 2003; Hu 2005): огрубление паросочетанием
   по тяжёлым рёбрам, укладка на грубейшем уровне, интерполяция обратно
   с дошлифовкой и адаптивным шагом Ху. Паросочетание не склеивает вершины
   разных сообществ: при слепом огрублении доля внутрикластерного веса рёбер
   падает с 0.85 до 0.30 (research/results/ablation.json).
2. Модель сил (1,-1) по Noack (как в ForceAtlas2): f_a(d) = w * d вдоль
   рёбер, f_r(d) = k^2 / d между парами, равновесие соседей k = sqrt(1/n).
   Фрухтерман-Рейнгольд (2,-1) отвергнут: квадратичное притяжение на длинных
   меж-кластерных рёбрах стягивает сообщества в "клубок" (Noack 2007: чем
   меньше разность показателей, тем лучше разделяются кластеры).
3. Гибридное отталкивание, O(n log n) на итерацию: ближнее поле — точно по
   сетке с ячейкой 2k (grid-вариант ФР), дальнее — выборка S вершин с
   масштабом n/S (Gove 2019; negative sampling в DRGraph). Без ближнего поля
   выборка недооценивает отталкивание соседей (веса ~1/d) и кластеры слипаются.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

COARSE_N = 400        # останавливаем огрубление, когда вершин меньше
EXACT_N = 1200        # до этого размера отталкивание считается точно
# Точная ветвь O(n^2) при n=1200 стоит 20.2 мс на итерацию против 2.7 мс
# у гибридной, но именно она задаёт глобальное расположение кластеров:
# при EXACT_N=400 разделение сообществ падает с 6.6 до 3.3 (выигрыш 0.7 с).
SAMPLE_S = 16         # размер случайной выборки для аппроксимации
BASE_ITER = 120       # итераций на самом грубом уровне
FINE_ITER = 25        # минимум итераций на самом тонком уровне
STEP_RATIO = 0.9      # коэффициент t адаптивного шага (Hu 2005)
GRAVITY = 0.05        # слабая гравитация: держит компоненты связности вместе


# ── огрубление ──────────────────────────────────────────────────────


def _heavy_edge_matching(
    n: int,
    edges: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    labels: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Каждая вершина сливается с ещё не сматченным соседом с максимальным
    весом ребра; при заданных labels — только внутри одного сообщества.
    Возвращает (число огрублённых вершин, отображение fine -> coarse)."""
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        if labels is not None and labels[u] != labels[v]:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))

    match = np.full(n, -1, dtype=np.int64)
    for v in rng.permutation(n):
        if match[v] != -1:
            continue
        best, best_w = -1, -1.0
        for u, w in adj[v]:
            if match[u] == -1 and u != v and w > best_w:
                best, best_w = u, w
        if best != -1:
            match[v] = best
            match[best] = v
        else:
            match[v] = v

    cid = np.full(n, -1, dtype=np.int64)
    nc = 0
    for v in range(n):
        if cid[v] == -1:
            cid[v] = nc
            partner = match[v]
            if partner != v and partner != -1:
                cid[partner] = nc
            nc += 1
    return nc, cid


def _coarsen_edges(
    nc: int, cid: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Переносит рёбра на грубый уровень, агрегируя кратные рёбра суммой весов."""
    cu = cid[edges[:, 0]]
    cv = cid[edges[:, 1]]
    keep = cu != cv
    a = np.minimum(cu[keep], cv[keep])
    b = np.maximum(cu[keep], cv[keep])
    key = a * nc + b
    uniq, inv = np.unique(key, return_inverse=True)
    w = np.zeros(len(uniq))
    np.add.at(w, inv, weights[keep])
    coarse_edges = np.stack([uniq // nc, uniq % nc], axis=1)
    return coarse_edges, w


# ── силы ────────────────────────────────────────────────────────────


def _near_repulsion(
    pos: np.ndarray,
    mass: np.ndarray,
    k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Точное отталкивание в ближнем поле по сетке с ячейкой 2k: вершины
    сортируются по ячейке, границы 9 соседних ячеек ищутся двоичным поиском,
    пары обрабатываются векторно. Число пар линейно по n при равновесной
    плотности ~4 вершины на ячейку, сортировка даёт O(n log n) на вызов."""
    n = len(pos)
    force = np.zeros_like(pos)
    cell = 2.0 * k

    gx = np.floor(pos[:, 0] / cell).astype(np.int64)
    gy = np.floor(pos[:, 1] / cell).astype(np.int64)
    gx -= gx.min()
    gy -= gy.min()
    stride = gy.max() + 3
    cid = gx * stride + gy

    order = np.argsort(cid, kind="stable")
    sorted_cid = cid[order]
    pair_cap = 48 * n  # защита от вырожденного случая (все в одной ячейке)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            target = cid + dx * stride + dy
            left = np.searchsorted(sorted_cid, target, side="left")
            right = np.searchsorted(sorted_cid, target, side="right")
            counts = right - left
            total = int(counts.sum())
            if total == 0:
                continue

            src = np.repeat(np.arange(n), counts)
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            tgt = order[np.repeat(left, counts) + offsets]

            scale = 1.0
            if total > pair_cap:
                sel = rng.integers(0, total, size=pair_cap)
                src, tgt = src[sel], tgt[sel]
                scale = total / pair_cap

            keep = src != tgt
            src, tgt = src[keep], tgt[keep]
            if len(src) == 0:
                continue

            delta = pos[src] - pos[tgt]
            dist2 = np.maximum(delta[:, 0] ** 2 + delta[:, 1] ** 2, 1e-9)
            coef = scale * (k * k) * mass[src] * mass[tgt] / dist2
            f = coef[:, None] * delta
            force[:, 0] += np.bincount(src, f[:, 0], minlength=n)
            force[:, 1] += np.bincount(src, f[:, 1], minlength=n)

    return force


def _forces(
    pos: np.ndarray,
    edges: np.ndarray,
    ew: np.ndarray,
    mass: np.ndarray,
    k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Суммарная сила на каждую вершину: притяжение по рёбрам (точно),
    отталкивание точно (n <= EXACT_N) или по случайной выборке."""
    n = len(pos)
    force = np.zeros_like(pos)

    # притяжение f_a = w * d вдоль ребра (модель (1,-1))
    if len(edges):
        delta = pos[edges[:, 1]] - pos[edges[:, 0]]
        fa = ew[:, None] * delta
        force[:, 0] += np.bincount(edges[:, 0], fa[:, 0], minlength=n)
        force[:, 1] += np.bincount(edges[:, 0], fa[:, 1], minlength=n)
        force[:, 0] -= np.bincount(edges[:, 1], fa[:, 0], minlength=n)
        force[:, 1] -= np.bincount(edges[:, 1], fa[:, 1], minlength=n)

    # отталкивание f_r = m_u*m_v*k^2/d: coef = k^2/d^2, модуль coef*delta = k^2/d
    if n <= EXACT_N:
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = np.maximum(np.einsum("ijk,ijk->ij", delta, delta), 1e-9)
        coef = (k * k) * np.outer(mass, mass) / dist2
        np.fill_diagonal(coef, 0.0)
        force += np.einsum("ij,ijk->ik", coef, delta)
    else:
        idx = rng.integers(0, n, size=(n, SAMPLE_S))
        delta = pos[:, None, :] - pos[idx]
        dist2 = np.maximum(np.einsum("ijk,ijk->ij", delta, delta), 1e-9)
        coef = (k * k) * (mass[:, None] * mass[idx]) / dist2
        force += (n / SAMPLE_S) * np.einsum("ij,ijk->ik", coef, delta)
        force += _near_repulsion(pos, mass, k, rng)

    center = np.average(pos, axis=0, weights=mass)
    force -= GRAVITY * (pos - center) * mass[:, None]
    return force


def _refine(
    pos: np.ndarray,
    edges: np.ndarray,
    ew: np.ndarray,
    mass: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Итерационная дошлифовка укладки с адаптивным шагом (Hu 2005)."""
    n = len(pos)
    k = np.sqrt(1.0 / max(n, 1))
    step = k * 0.5
    prev_energy = np.inf
    progress = 0

    for _ in range(iters):
        force = _forces(pos, edges, ew, mass, k, rng)
        norms = np.maximum(np.hypot(force[:, 0], force[:, 1]), 1e-9)
        pos = pos + step * force / norms[:, None]

        energy = float(np.sum(norms * norms))
        if energy < prev_energy:
            progress += 1
            if progress >= 5:
                progress = 0
                step /= STEP_RATIO
        else:
            progress = 0
            step *= STEP_RATIO
        prev_energy = energy
    return pos


# ── основной алгоритм ───────────────────────────────────────────────


def multilevel_layout(
    n: int,
    edges: np.ndarray,
    seed: int = 42,
    labels: np.ndarray | None = None,
    debug_hook=None,
) -> np.ndarray:
    """Укладка n вершин с рёбрами edges (m x 2) -> координаты n x 2
    (ненормализованные). labels — метки сообществ для огрубления,
    debug_hook(level, n_level, pos, edges, labels) — после каждого уровня."""
    rng = np.random.default_rng(seed)

    if n == 0:
        return np.zeros((0, 2))
    if len(edges):
        edges = edges[edges[:, 0] != edges[:, 1]]

    weights = np.ones(len(edges))
    mass = np.ones(n)

    # фаза 1: иерархия огрублений
    hierarchy: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    cur_n, cur_edges, cur_w, cur_mass = n, edges, weights, mass
    cur_labels = labels
    while cur_n > COARSE_N and len(cur_edges) > 0:
        nc, cid = _heavy_edge_matching(cur_n, cur_edges, cur_w, rng, cur_labels)
        if nc >= cur_n * 0.95:  # огрубление застопорилось
            break
        coarse_edges, coarse_w = _coarsen_edges(nc, cid, cur_edges, cur_w)
        coarse_mass = np.zeros(nc)
        np.add.at(coarse_mass, cid, cur_mass)
        hierarchy.append((cur_n, cur_edges, cur_w, cur_mass, cid, cur_labels))
        if cur_labels is not None:
            coarse_labels = np.zeros(nc, dtype=np.int64)
            coarse_labels[cid] = cur_labels  # внутри пары метки совпадают
            cur_labels = coarse_labels
        cur_n, cur_edges, cur_w, cur_mass = nc, coarse_edges, coarse_w, coarse_mass

    # фаза 2: старт со случайной укладки на грубейшем уровне
    pos = rng.random((cur_n, 2))
    pos = _refine(pos, cur_edges, cur_w, cur_mass, BASE_ITER, rng)
    if debug_hook:
        debug_hook(len(hierarchy), cur_n, pos, cur_edges, cur_labels)

    # фаза 3: интерполяция вверх с дошлифовкой на каждом уровне
    iters = BASE_ITER
    for level, (fine_n, fine_edges, fine_w, fine_mass, cid, fine_labels) in enumerate(
        reversed(hierarchy)
    ):
        k_fine = np.sqrt(1.0 / fine_n)
        pos = pos[cid] + rng.normal(0.0, k_fine * 0.3, size=(fine_n, 2))
        iters = max(FINE_ITER, int(iters * 0.7))
        pos = _refine(pos, fine_edges, fine_w, fine_mass, iters, rng)
        if debug_hook:
            debug_hook(len(hierarchy) - 1 - level, fine_n, pos, fine_edges, fine_labels)

    return pos


def fast_layout(
    G: nx.Graph, seed: int = 42, communities: dict | None = None
) -> dict[int, list[float]]:
    """Обёртка для nx.Graph: {узел: [x, y]}, координаты ненормализованные.
    communities — {узел: сообщество} для информированного огрубления."""
    nodes = list(G.nodes())
    index = {nd: i for i, nd in enumerate(nodes)}
    if G.number_of_edges():
        edges = np.array(
            [[index[u], index[v]] for u, v in G.edges()], dtype=np.int64
        )
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
    labels = None
    if communities is not None:
        labels = np.zeros(len(nodes), dtype=np.int64)
        for nd, i in index.items():
            labels[i] = communities.get(nd, -1)
    pos = multilevel_layout(len(nodes), edges, seed=seed, labels=labels)
    return {nd: [float(pos[i, 0]), float(pos[i, 1])] for nd, i in index.items()}
