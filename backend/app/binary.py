"""Бинарная сериализация графа для передачи на клиент.

JSON графа из 18 772 вершин и 198 050 рёбер — 13.0 МБ, 0.48 с и 216 822 объекта
Pydantic, плюс столько же объектов JS на клиенте. Плоские массивы — 3.1 МБ и
0.3 мс, читаются без аллокаций в Float32Array/Uint32Array для WebGL и воркера.

Формат (little-endian):
    magic       4 байта  b"SFG1"
    hdr_len     uint32   длина JSON-заголовка
    header      hdr_len  UTF-8 JSON: метрики, топология, размеры массивов
    arrays      подряд, в порядке, объявленном в header["arrays"]

Рёбра — ИНДЕКСЫ вершин, не идентификаторы: иначе клиенту 198 050 поисков в словаре.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"SFG1"

# Заголовок дополняется пробелами до границы в 8 байт: иначе начало массивов
# зависит от длины JSON, а Int32Array не создаётся на невыровненном смещении.
ALIGN = 8


def _pack(header: dict, buffers: list[bytes]) -> bytes:
    hdr = json.dumps(header, ensure_ascii=False).encode("utf-8")
    pad = (-(8 + len(hdr))) % ALIGN
    hdr += b" " * pad
    return b"".join([MAGIC, struct.pack("<I", len(hdr)), hdr, *buffers])

NODE_ARRAYS = [
    ("id", "<i4"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("degree", "<u4"),
    ("community", "<u4"),
    ("degree_centrality", "<f4"),
    ("betweenness_centrality", "<f4"),
    ("closeness_centrality", "<f4"),
    ("eigenvector_centrality", "<f4"),
    ("member_count", "<u4"),
    ("is_collapsed", "<u1"),
]

EDGE_ARRAYS = [
    ("source", "<u4"),  # индекс вершины, не идентификатор
    ("target", "<u4"),
    ("weight", "<f4"),
]


def encode_arrays(
    columns: dict[str, np.ndarray],
    *,
    n: int,
    m: int,
    global_metrics: dict,
    topology: str,
    recommended_layout: str,
    community_sizes: dict[int, int],
) -> bytes:
    """Кодирует граф из готовых массивов numpy, без промежуточных словарей."""
    buffers: list[bytes] = []
    layout: list[dict] = []
    offset = 0

    for name, dtype in NODE_ARRAYS + EDGE_ARRAYS:
        count = n if (name, dtype) in NODE_ARRAYS else m
        col = columns[name]
        raw = np.ascontiguousarray(col, dtype=np.dtype(dtype)).tobytes()
        buffers.append(raw)
        layout.append({"name": name, "dtype": dtype, "count": count, "offset": offset})
        offset += len(raw)
        # иначе is_collapsed (1 байт) сдвинет массивы на n и сломает Uint32Array
        pad = (-offset) % ALIGN
        if pad:
            buffers.append(b"\0" * pad)
            offset += pad

    header = {
        "n": n,
        "m": m,
        "topology": topology,
        "recommended_layout": recommended_layout,
        "num_communities": len(community_sizes),
        "community_sizes": {str(k): v for k, v in community_sizes.items()},
        "global_metrics": global_metrics,
        "arrays": layout,
    }
    return _pack(header, buffers)


def encode_graph(
    nodes: list[dict],
    edges: list[dict],
    *,
    global_metrics: dict,
    topology: str,
    recommended_layout: str,
    community_sizes: dict[int, int],
) -> bytes:
    """Кодирует граф из списков словарей; source/target рёбер приходят
    идентификаторами вершин и переводятся здесь в индексы."""
    n = len(nodes)
    m = len(edges)
    index_of = {int(nd["id"]): i for i, nd in enumerate(nodes)}

    cols: dict[str, np.ndarray] = {}
    for name, dtype in NODE_ARRAYS:
        if name == "is_collapsed":
            cols[name] = np.fromiter(
                (1 if nd.get("is_collapsed") else 0 for nd in nodes), np.uint8, n
            )
        elif name == "member_count":
            cols[name] = np.fromiter(
                (int(nd.get("member_count", 1)) for nd in nodes), np.uint32, n
            )
        else:
            cols[name] = np.fromiter((nd[name] for nd in nodes), np.dtype(dtype), n)

    cols["source"] = np.fromiter((index_of[int(e["source"])] for e in edges), np.uint32, m)
    cols["target"] = np.fromiter((index_of[int(e["target"])] for e in edges), np.uint32, m)
    cols["weight"] = np.fromiter(
        (float(e.get("weight", 1) or 1) for e in edges), np.float32, m
    )

    return encode_arrays(
        cols, n=n, m=m,
        global_metrics=global_metrics,
        topology=topology,
        recommended_layout=recommended_layout,
        community_sizes=community_sizes,
    )


def encode_focus(
    node_ids: list[int],
    opacities: dict[int, float],
    context_nodes: list[int],
    focus_node: int,
) -> bytes:
    """Кодирует результат фокусировки: прозрачность на вершину.

    Прозрачности рёбер НЕ передаются, клиент выводит их из концевых вершин:
    экономит ~4 МБ на 198 050 рёбрах и столько же строковых ключей "u-v".
    Идентификаторы идут рядом: клиент может показывать не тот набор вершин,
    для которого посчитан фокус (свёрнутый граф), сопоставление по id."""
    n = len(node_ids)
    ids = np.fromiter((int(nid) for nid in node_ids), dtype=np.int32, count=n)
    alpha = np.fromiter(
        (opacities.get(int(nid), 0.06) for nid in node_ids), dtype=np.float32, count=n
    )
    header = {
        "n": n,
        "focus_node": focus_node,
        "context_size": len(context_nodes),
        "arrays": [
            {"name": "id", "dtype": "<i4", "count": n, "offset": 0},
            {"name": "opacity", "dtype": "<f4", "count": n, "offset": 4 * n},
        ],
    }
    return _pack(header, [ids.tobytes(), alpha.tobytes()])
