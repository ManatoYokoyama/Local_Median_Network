import numpy as np
import pandas as pd
import csv
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import os
from pathlib import Path
from sklearn.manifold import MDS
import matplotlib as mpl
mpl.use("TkAgg")

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

mpl.rcParams['figure.dpi'] = 150      # 画面表示用の解像度


# =========================
# Logging utilities (for figure-friendly step logs)
# =========================

@dataclass
class LogEvent:
    t: float
    step: str
    data: dict[str, Any]


def _safe(obj: Any) -> Any:
    """JSON-serializable へ寄せるための安全変換。"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    # numpy types
    try:
        import numpy as _np  # local
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def make_step_logger(output_dir: str, base_name: str) -> tuple[Callable[..., None], list[LogEvent], Callable[[], None]]:
    """擬似コードの各ステップ名でログを出す（stdout + JSONL + TXT）。"""
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{base_name}_method2_log.jsonl")
    txt_path = os.path.join(output_dir, f"{base_name}_method2_log.txt")

    events: list[LogEvent] = []
    lines: list[str] = []

    def log(step: str, **data: Any) -> None:
        ev = LogEvent(t=time.time(), step=str(step), data={k: _safe(v) for k, v in data.items()})
        events.append(ev)

        payload = ", ".join([f"{k}={ev.data[k]}" for k in sorted(ev.data.keys())])
        line = f"[{ev.step}] {payload}" if payload else f"[{ev.step}]"
        print(line)
        lines.append(line)

    def flush() -> None:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps({"t": ev.t, "step": ev.step, "data": ev.data}, ensure_ascii=False) + "\n")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[LOG] wrote: {jsonl_path}")
        print(f"[LOG] wrote: {txt_path}")

    return log, events, flush


# === 新規追加: SplitsTree .dist (NEXUS: TAXA + DISTANCES) 読み込み ===
def read_dist_nexus(file_path: str) -> tuple[np.ndarray, list[str]]:
    """SplitsTree の .dist (NEXUS: TAXA + DISTANCES) を読み、(matrix, vertices) を返す。

    期待する形式（代表例）:
      - BEGIN TAXA; ... TAXLABELS ... ; END;
      - BEGIN DISTANCES; ... MATRIX
            A 0 1 2 ...
            B 1 0 3 ...
        ; END;

    LABELS=LEFT を想定し、各行の先頭トークンを taxon 名として捨てる。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    # --- parse TAXLABELS ---
    taxa: list[str] = []
    in_taxlabels = False
    for ln in raw_lines:
        u = ln.upper()
        if u.startswith("TAXLABELS"):
            in_taxlabels = True
            continue
        if in_taxlabels:
            if ln.endswith(";"):
                ln2 = ln[:-1].strip()
                if ln2:
                    taxa.append(_strip_nexus_token(ln2))
                break
            if ln:
                taxa.append(_strip_nexus_token(ln))

    # --- parse MATRIX ---
    matrix_rows: list[list[float]] = []
    in_matrix = False
    for ln in raw_lines:
        u = ln.upper()
        if u.startswith("MATRIX"):
            in_matrix = True
            continue
        if in_matrix:
            if ln.endswith(";"):
                ln = ln[:-1].strip()
                if not ln:
                    break
                # fallthrough to parse last row
                end_after = True
            else:
                end_after = False

            if not ln:
                if end_after:
                    break
                continue

            parts = ln.split()
            if len(parts) <= 1:
                if end_after:
                    break
                continue

            # 先頭は taxon 名（LABELS=LEFT）として捨てる
            row_vals = [float(x) for x in parts[1:]]
            matrix_rows.append(row_vals)

            if end_after:
                break

    if not taxa:
        raise ValueError(f"No TAXLABELS found in .dist file: {file_path}")
    if not matrix_rows:
        raise ValueError(f"No MATRIX found in .dist file: {file_path}")

    mat = np.array(matrix_rows, dtype=float)
    if mat.shape[0] != len(taxa):
        # TAXLABELS と MATRIX の行数がズレるケースもあるので、MATRIX 側の行数に合わせる
        taxa = taxa[: mat.shape[0]]
    if mat.shape[1] != len(taxa):
        # 末尾欠けなどを補正できないのでエラー
        raise ValueError(
            f"MATRIX width ({mat.shape[1]}) != NTAX ({len(taxa)}). file={file_path}"
        )
    return mat, taxa

# --- 新規追加: 上位K本の長い辺を削除（削除ごとにbridgeを再判定） ---
def prune_long_edges_by_count(G: nx.Graph, edge_prune_k: int) -> tuple[nx.Graph, int, float | None]:
    """
    辺長の上位K本（weight が大きい順）を削除する。

    安全策:
      - 各削除のたびに bridge を再計算し、連結成分が増える削除を避ける。
      - 端点が次数1になる削除も避ける。

    戻り値: (G, removed_edge_count, threshold_used)
      threshold_used は「候補（重み降順）のうち、K本目の重み（参考）」。
      何も削除しない場合は None。
    """
    if edge_prune_k is None:
        return G, 0, None
    try:
        K = int(edge_prune_k)
    except Exception:
        return G, 0, None
    if K <= 0:
        return G, 0, None

    if G.number_of_edges() == 0:
        return G, 0, None

    edges_all = [(u, v, float(data.get("weight", 1.0))) for u, v, data in G.edges(data=True)]
    edges_all.sort(key=lambda x: x[2], reverse=True)

    K_eff = min(K, len(edges_all))
    thr = edges_all[K_eff - 1][2] if K_eff > 0 else None

    removed = 0
    for u, v, w in edges_all:
        if removed >= K:
            break
        if not G.has_edge(u, v):
            continue
        if G.degree[u] <= 1 or G.degree[v] <= 1:
            continue

        bridges = set(nx.bridges(G)) if G.number_of_edges() > 0 else set()
        if (u, v) in bridges or (v, u) in bridges:
            continue

        G.remove_edge(u, v)
        removed += 1

    if removed == 0:
        return G, 0, None
    return G, removed, thr

def _strip_nexus_token(tok: str) -> str:
    tok = tok.strip()
    # SplitsTree6 では 'A' のように quote されることがある
    if (len(tok) >= 2) and ((tok[0] == "'" and tok[-1] == "'") or (tok[0] == '"' and tok[-1] == '"')):
        return tok[1:-1]
    return tok


def add_edge_to_Neighborhood(G, vertices, matrix, Neighborhood, vertex1, logger: Optional[Callable[..., None]] = None):
    for vertex_B in Neighborhood:
        index_A = vertices.index(vertex1)
        index_B = vertices.index(vertex_B)
        D_AB = calculate_distances(matrix, (index_A, index_B))
        G.add_edge(vertex1, vertex_B, weight=D_AB)
        if logger is not None:
            logger(
                "AddEdge {a,b} (Neighborhood)",
                a=vertex1,
                b=vertex_B,
                ell=D_AB,
            )
        else:
            print("Neighborhood add edge:", vertex1, "-", vertex_B, "weight=", D_AB)

def add_new_vertex(G, vertices, matrix, Bunch, Bunch_Base, vertex1, count, rate, rate_XD, min_distance, deadline: float | None = None, logger: Optional[Callable[..., None]] = None):
    index_of = {v: i for i, v in enumerate(vertices)}
    if logger is not None:
        logger(
            "For B in 𝓑(a): using Base(B)",
            a=vertex1,
            B=list(Bunch),
            Base=list(Bunch_Base),
        )
    else:
        print("房基底:", Bunch_Base)
    for vertexB in Bunch_Base:
        a_min = float('inf')
        vertex2 = None
        vertex3 = None
        for vertexC in Bunch:
            if vertexC == vertexB:
                continue
            a, _ = calculate_a_b(matrix, vertices, vertex1, vertexB, vertexC, rate, min_distance, index_of = index_of)
            if 0 < a < a_min:
                a_min = a
                vertex2 = vertexB
                vertex3 = vertexC
                print("vertex2:",vertex2,"vertex3:",vertex3)
        if vertex2 is None or vertex3 is None or (not np.isfinite(a_min)):
            if logger is not None:
                logger(
                    "continue (no c with m1(a;x,c)>0)",
                    a=vertex1,
                    x=vertexB,
                    B=list(Bunch),
                )
            continue
        matrix, vertices, new_vertex_name, vertex4, XD_min = add_new_vertex_Bunch(
            matrix, vertices, vertex1, vertex2, vertex3, a_min, count, deadline=deadline, logger=logger)
        index_A = vertices.index(vertex1)
        index_B = vertices.index(vertex2)
        index_C = vertices.index(vertex3)
        index_D = vertices.index(vertex4)
        index_X = vertices.index(new_vertex_name)
        D_AX = calculate_distances(matrix, (index_A, index_X))
        D_BX = calculate_distances(matrix, (index_B, index_X))
        D_CX = calculate_distances(matrix, (index_C, index_X))
        if logger is not None:
            logger(
                "Compute α and XD_min",
                a=vertex1,
                x=vertex2,
                c=vertex3,
                alpha=a_min,
                d_star=vertex4,
                XD_min=XD_min,
                D_ax=D_AX,
                D_bx=D_BX,
                D_cx=D_CX,
                rate_XD=rate_XD,
            )
        if rate_XD * XD_min < min(D_AX, D_BX, D_CX):
            if logger is not None:
                logger(
                    "Decision: add edge {a,d*} (skip new median)",
                    a=vertex1,
                    d_star=vertex4,
                    criterion="rate_XD*XD_min < min(D_ax,D_bx,D_cx)",
                    lhs=rate_XD * XD_min,
                    rhs=min(D_AX, D_BX, D_CX),
                )
            #グラフに追加した新頂点を削除し，A-Dを辺で結ぶ
            matrix, vertices = remove_vertex(matrix, vertices, new_vertex_name)
            D_AD = calculate_distances(matrix, (index_A, index_D))
            G.add_edge(vertex1, vertex4, weight=D_AD)
            if logger is not None:
                logger(
                    "AddEdge {a,d*}",
                    a=vertex1,
                    d_star=vertex4,
                    ell=D_AD,
                )
        else:
            if logger is not None:
                logger(
                    "Decision: introduce new median vertex m",
                    a=vertex1,
                    m=new_vertex_name,
                    alpha=a_min,
                    d_star=vertex4,
                    criterion="rate_XD*XD_min >= min(D_ax,D_bx,D_cx)",
                    lhs=rate_XD * XD_min,
                    rhs=min(D_AX, D_BX, D_CX),
                )
            # グラフに新しい頂点を追加し、vertex1と新頂点を辺で結ぶ
            G.add_node(new_vertex_name)
            G.add_edge(vertex1, new_vertex_name, weight=a_min)
            if logger is not None:
                logger(
                    "AddEdge {a,m}",
                    a=vertex1,
                    m=new_vertex_name,
                    ell=a_min,
                )
            count += 1
    return matrix, vertices, count

def add_new_vertex_Bunch(matrix, vertices, vertexA, vertexB, vertexC, a_min, count, deadline: float | None = None, logger: Optional[Callable[..., None]] = None):
    index_A = vertices.index(vertexA)
    index_B = vertices.index(vertexB)
    index_C = vertices.index(vertexC)
    D_AB = calculate_distances(matrix, (index_A, index_B))
    D_CA = calculate_distances(matrix, (index_A, index_C))
    new_vertex_name = f"m{count}"
    new_row = np.zeros((1, matrix.shape[1]))
    new_column = np.zeros((matrix.shape[0] + 1, 1))
    new_matrix = np.vstack([matrix, new_row])
    new_matrix = np.hstack([new_matrix, new_column])
    new_matrix[-1, index_A] = a_min
    new_matrix[index_A, -1] = a_min
    if logger is not None:
        logger(
            "Extend m on V∪{m}: set distances to a,x,c",
            a=vertexA,
            x=vertexB,
            c=vertexC,
            m=new_vertex_name,
            alpha=a_min,
            D_ab=D_AB,
            D_ca=D_CA,
            m_x=D_AB - a_min,
            m_c=D_CA - a_min,
        )
    else:
        print("a_min:", a_min, "D_AB", D_AB, "D_CA:", D_CA)
    new_matrix[-1, index_B] = D_AB - a_min
    new_matrix[index_B, -1] = D_AB - a_min
    # print(vertexB,"-",new_vertex_name,"weight=",D_AB-a_min)
    new_matrix[-1, index_C] = D_CA - a_min
    new_matrix[index_C, -1] = D_CA - a_min
    # print(vertexC,"-",new_vertex_name,"weight=",D_CA-a_min)
    vertices.append(new_vertex_name)
    vertices_D = vertices.copy()
    for v in [vertexA, vertexB, vertexC, new_vertex_name]:
        vertices_D.remove(v)
    XD_min = float('inf')
    vertexX = None
    for vertexD in vertices_D:
        _check_timeout(deadline)
        D_XD = XD_calculation(matrix, vertices, vertexA, vertexB, vertexC, vertexD)
        if logger is not None:
            logger(
                "Compute XD(a,x,c;d)",
                a=vertexA,
                x=vertexB,
                c=vertexC,
                d=vertexD,
                XD=D_XD,
            )
        if D_XD < XD_min:
            XD_min = D_XD
            vertexX = vertexD
        index_D = vertices.index(vertexD)
        new_matrix[-1, index_D] = D_XD
        new_matrix[index_D, -1] = D_XD
        # print(vertexD,"-",new_vertex_name,"weight=",D_XD)
    if logger is not None:
        logger(
            "XD minimizer d*",
            a=vertexA,
            x=vertexB,
            c=vertexC,
            m=new_vertex_name,
            d_star=vertexX,
            XD_min=XD_min,
        )
    return new_matrix, vertices, new_vertex_name, vertexX, XD_min

def calculate_a_b(matrix, vertices, vertexA, vertexB, vertexC, rate, min_distance, index_of=None):
    """
    a = (D_AB - D_BC + D_CA) / 2
    b = (D_AB + D_BC - D_CA) / 2
    """
    if index_of is None:
        index_A = vertices.index(vertexA)
        index_B = vertices.index(vertexB)
        index_C = vertices.index(vertexC)
    else:
        index_A = index_of[vertexA]
        index_B = index_of[vertexB]
        index_C = index_of[vertexC]
    
    D_AB = calculate_distances(matrix, (index_A, index_B))
    D_BC = calculate_distances(matrix, (index_B, index_C))
    D_CA = calculate_distances(matrix, (index_C, index_A))
    
    a = (D_AB - D_BC + D_CA) / 2
    b = (D_AB + D_BC - D_CA) / 2
    if rate * a < min_distance:
        a = 0
    if rate * b < min_distance:
        b = 0
    return a, b

def calculate_distances(matrix, vertex_indices):
    return matrix[vertex_indices]

def classify_surrounding_vertices(vertices, matrix, surrounding_vertices, vertex1, rate, min_distance, logger: Optional[Callable[..., None]] = None):
    index_of = {v: i for i, v in enumerate(vertices)}
    # a=0となる直結点を記録するリスト
    Neighborhood = []
    # a!=0となる近接点を記録するリスト     
    indirect_vertices = []
    if logger is not None:
        logger(
            "(Surrounding) input C(a)",
            a=vertex1,
            C_a=list(surrounding_vertices),
            mode=f"rate={rate}",
            min_distance=min_distance,
        )
    else:
        print("周辺点:", surrounding_vertices)
    if len(surrounding_vertices) >= 2:
        # 周辺点からCの候補を見つけるリスト
        for vertex_B in surrounding_vertices:
            candidates_C = surrounding_vertices.copy()
            candidates_C.remove(vertex_B)
            a_values = [calculate_a_b(matrix, vertices, vertex1, vertex_B, vertex_C, rate, min_distance, index_of = index_of)[0]
                        for vertex_C in candidates_C]
            if all(a == 0 for a in a_values):
                Neighborhood.append(vertex_B)
            else:
                indirect_vertices.append(vertex_B)
    elif len(surrounding_vertices) == 1:
        Neighborhood = surrounding_vertices.copy()
        print("近傍:",Neighborhood)
    if logger is not None:
        logger(
            "Split S(a)=N(a) ⊔ I(a)",
            a=vertex1,
            S_a=list(surrounding_vertices),
            N_a=list(Neighborhood),
            I_a=list(indirect_vertices),
        )
    return Neighborhood, indirect_vertices

def _triangle_decompose_nonnegative(d_ab: float, d_bc: float, d_ca: float, tol: float = 1e-12) -> tuple[float, float, float]:
    """\
    Given triangle edge lengths (d_ab, d_bc, d_ca), compute nonnegative limb lengths (alpha,beta,gamma)
    such that ideally:
        alpha+beta = d_ab
        beta+gamma = d_bc
        gamma+alpha = d_ca

    If the triangle inequalities are violated (or due to numerical noise), the exact solution may
    contain negatives. In that case, return the best (least-squares) solution under constraints
    alpha,beta,gamma >= 0 by enumerating active sets.

    This prevents creation of negative-weight edges during triangle collapse.
    """
    # Unconstrained exact solution
    alpha0 = 0.5 * (d_ab + d_ca - d_bc)
    beta0  = 0.5 * (d_ab + d_bc - d_ca)
    gamma0 = 0.5 * (d_ca + d_bc - d_ab)

    # If only tiny negatives due to floating error, clip.
    if (alpha0 >= -tol) and (beta0 >= -tol) and (gamma0 >= -tol):
        return (max(0.0, float(alpha0)), max(0.0, float(beta0)), max(0.0, float(gamma0)))

    # Least squares under nonnegativity by active-set enumeration.
    # Equations: A x ~= b
    A = np.array(
        [[1.0, 1.0, 0.0],
         [0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0]],
        dtype=float
    )
    b = np.array([float(d_ab), float(d_bc), float(d_ca)], dtype=float)

    best_x = None
    best_res = float('inf')

    # Enumerate which variables are fixed to 0
    # mask[i]=True means variable i is fixed (alpha/beta/gamma) to 0.
    for mask in (
        (False, False, False),
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
        (True,  True,  False),
        (True,  False, True),
        (False, True,  True),
        (True,  True,  True),
    ):
        fixed = np.array(mask, dtype=bool)
        free_idx = np.where(~fixed)[0]

        x = np.zeros(3, dtype=float)
        # if all fixed, x=0
        if free_idx.size > 0:
            Af = A[:, free_idx]
            bf = b.copy()  # since fixed vars are 0, no need to subtract
            # Solve least squares Af * x_free ~= bf
            x_free, *_ = np.linalg.lstsq(Af, bf, rcond=None)
            x[free_idx] = x_free

        # Feasibility: nonnegative within tolerance
        if np.any(x < -tol):
            continue
        x = np.maximum(x, 0.0)

        res = float(np.linalg.norm(A @ x - b))
        if res < best_res:
            best_res = res
            best_x = x

    if best_x is None:
        # Fallback: simple clipping of unconstrained solution
        return (max(0.0, float(alpha0)), max(0.0, float(beta0)), max(0.0, float(gamma0)))

    return (float(best_x[0]), float(best_x[1]), float(best_x[2]))

def collapse_triangles_with_median(G: nx.Graph, count: int):
    nodes = list(G.nodes())
    triangles = []
    for a, b, c in itertools.combinations(nodes, 3):
        if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a):
            triangles.append((a, b, c))
    for a, b, c in triangles:
        if not (G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a)):
            continue
        d_ab = G[a][b]['weight']
        d_bc = G[b][c]['weight']
        d_ca = G[c][a]['weight']
        alpha, beta, gamma = _triangle_decompose_nonnegative(d_ab, d_bc, d_ca)
        while True:
            new_name = f"m{count}"
            count += 1
            if new_name not in G:
                break
        G.add_node(new_name)
        G.add_edge(new_name, a, weight=max(0.0, alpha))
        G.add_edge(new_name, b, weight=max(0.0, beta))
        G.add_edge(new_name, c, weight=max(0.0, gamma))
        if G.has_edge(a, b):
            G.remove_edge(a, b)
        if G.has_edge(b, c):
            G.remove_edge(b, c)
        if G.has_edge(c, a):
            G.remove_edge(c, a)
    return G, count


# --- 新規追加: collapse_triangles_touching_edges ---
def collapse_triangles_touching_edges(G: nx.Graph, count: int, focus_edges: set[tuple[str, str]]):
    """\
    focus_edges に含まれる辺（無向なので (min,max) で正規化）に接する三角形だけを潰す。
    """
    if not focus_edges:
        return G, count

    def norm(u: str, v: str) -> tuple[str, str]:
        return (u, v) if u <= v else (v, u)

    focus = {norm(u, v) for (u, v) in focus_edges}

    nodes = list(G.nodes())
    for a, b, c in itertools.combinations(nodes, 3):
        if not (G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a)):
            continue
        # triangle edges
        e1 = norm(a, b)
        e2 = norm(b, c)
        e3 = norm(c, a)
        if (e1 not in focus) and (e2 not in focus) and (e3 not in focus):
            continue

        # (a,b,c) は focus edge に触れている三角形
        d_ab = G[a][b]['weight']
        d_bc = G[b][c]['weight']
        d_ca = G[c][a]['weight']
        alpha, beta, gamma = _triangle_decompose_nonnegative(d_ab, d_bc, d_ca)

        while True:
            new_name = f"m{count}"
            count += 1
            if new_name not in G:
                break
        G.add_node(new_name)
        G.add_edge(new_name, a, weight=max(0.0, alpha))
        G.add_edge(new_name, b, weight=max(0.0, beta))
        G.add_edge(new_name, c, weight=max(0.0, gamma))
        if G.has_edge(a, b):
            G.remove_edge(a, b)
        if G.has_edge(b, c):
            G.remove_edge(b, c)
        if G.has_edge(c, a):
            G.remove_edge(c, a)

    return G, count


# --- 新規追加: 長すぎる辺の削除（bridgeは削除しない） ---
def prune_long_edges(G: nx.Graph, rate_edge: float, base_len: float) -> tuple[nx.Graph, int]:
    """
    長すぎる辺（weight > rate_edge * base_len）を削除する。

    安全策:
      - 各削除のたびに bridge を再計算し、連結成分が増える削除を避ける。
      - 端点が次数1になる削除も避ける。

    戻り値は (G, removed_edge_count)。
    """
    if rate_edge is None:
        return G, 0
    try:
        r = float(rate_edge)
    except Exception:
        return G, 0
    if r <= 0:
        return G, 0
    if base_len is None:
        return G, 0
    if not np.isfinite(base_len) or base_len <= 0:
        return G, 0

    thr = r * float(base_len)
    if not np.isfinite(thr) or thr <= 0:
        return G, 0

    removed = 0

    edges = [(u, v, float(data.get("weight", 1.0))) for u, v, data in G.edges(data=True)]
    edges.sort(key=lambda x: x[2], reverse=True)

    for u, v, w in edges:
        if w <= thr:
            break
        if not G.has_edge(u, v):
            continue
        if G.degree[u] <= 1 or G.degree[v] <= 1:
            continue

        bridges = set(nx.bridges(G)) if G.number_of_edges() > 0 else set()
        if (u, v) in bridges or (v, u) in bridges:
            continue

        G.remove_edge(u, v)
        removed += 1

    return G, removed

# --- 新規追加: 上位 p% の長い辺を削除（bridgeは削除しない） ---
def prune_long_edges_by_percentile(G: nx.Graph, rate_edge_pct: float) -> tuple[nx.Graph, int, float | None]:
    """\
    辺長の上位 p%（weight が大きい順）を削除する。

    - bridge（削除すると連結成分が増える辺）は削除しない。
    - 端点が次数1になる削除も避ける（安全策）。

    戻り値: (G, removed_edge_count, threshold_used)
      threshold_used は削除対象とみなした最小の weight（= 分位点しきい値）。
      何も削除しない場合は None。
    """
    if rate_edge_pct is None:
        return G, 0, None
    try:
        p = float(rate_edge_pct)
    except Exception:
        return G, 0, None
    if p <= 0:
        return G, 0, None
    if p >= 100:
        p = 100.0

    m = G.number_of_edges()
    if m == 0:
        return G, 0, None

    # 何本削除するか（ceil で最低1本になりやすいので、まずは floor）
    k = int(np.floor(m * p / 100.0))
    if k <= 0:
        return G, 0, None

    # Bridges are edges that must not be removed to keep current connectivity.
    bridges = set(nx.bridges(G)) if G.number_of_edges() > 0 else set()

    edges = [(u, v, float(data.get('weight', 1.0))) for u, v, data in G.edges(data=True)]
    edges.sort(key=lambda x: x[2], reverse=True)

    # 分位点しきい値（参考用）: 上位k本のうち最小の weight
    thr = edges[k - 1][2] if (k - 1) < len(edges) else edges[-1][2]

    removed = 0
    for u, v, w in edges:
        if removed >= k:
            break
        if not G.has_edge(u, v):
            continue
        if (u, v) in bridges or (v, u) in bridges:
            continue
        if G.degree[u] <= 1 or G.degree[v] <= 1:
            continue
        G.remove_edge(u, v)
        removed += 1

    if removed == 0:
        return G, 0, None
    return G, removed, thr

def compute_mae(filtered_distance_matrix_df: pd.DataFrame, csv_distance_matrix_df: pd.DataFrame) -> float:
    error_df = (filtered_distance_matrix_df - csv_distance_matrix_df).abs()
    non_diag_error_df = error_df.mask(np.eye(error_df.shape[0], dtype=bool))
    mae = non_diag_error_df.stack().mean()
    return mae

def divide_distance_matrices(filtered_df, csv_df):
    result = filtered_df.divide(csv_df).replace([np.inf, -np.inf], np.nan)
    np.fill_diagonal(result.values, np.diag(filtered_df.values))
    return result

def filter_alphabetic_indices(distance_matrix_df):
    idx = distance_matrix_df.index.astype(str)
    cols = distance_matrix_df.columns.astype(str)
    if idx.str.match(r'^[A-Za-z]+$').all() and cols.str.match(r'^[A-Za-z]+$').all():
        return distance_matrix_df.loc[
            idx.str.match(r'^[A-Za-z]+$'),
            cols.str.match(r'^[A-Za-z]+$')
        ]
    else:
        return distance_matrix_df

def find_all_Bunches(vertices, matrix, indirect_vertices, vertex1, rate, min_distance, logger: Optional[Callable[..., None]] = None):
    index_of = {v: i for i, v in enumerate(vertices)}
    adj = {v: [] for v in indirect_vertices}
    k = len(indirect_vertices)
    for i in range(k):
        u = indirect_vertices[i]
        for j in range(i + 1, k):
            v = indirect_vertices[j]
            a, _ = calculate_a_b(
                matrix, vertices,
                vertex1, u, v,
                rate, min_distance,
                index_of=index_of
            )
            if a != 0:
                adj[u].append(v)
                adj[v].append(u)
    if logger is not None:
        m_edges = sum(len(vs) for vs in adj.values()) // 2
        logger(
            "Build H_a=(I(a),E_a)",
            a=vertex1,
            I_a=list(indirect_vertices),
            **{"|I_a|": len(indirect_vertices), "|E_a|": m_edges},
        )
    visited = set()
    all_Bunches = []
    for start in indirect_vertices:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    stack.append(y)
        comp.sort(key=lambda v: indirect_vertices.index(v))
        all_Bunches.append(comp)
    if logger is not None:
        logger(
            "Bunches 𝓑(a) = CC(H_a)",
            a=vertex1,
            bunches=[list(b) for b in all_Bunches],
            bunch_count=len(all_Bunches),
        )
    return all_Bunches

def find_Bunch_Base(vertices, matrix, Bunch, vertex1, rate, min_distance, logger: Optional[Callable[..., None]] = None):
    index_of = {v: i for i, v in enumerate(vertices)} 
    if logger is not None:
        logger(
            "(Base selection) start",
            a=vertex1,
            B=list(Bunch),
        )
    Bunch_Base = []
    unexplored = Bunch.copy()
    all_a_min = float('inf')
    a_min_vertex = None
    while unexplored:
        max_orthogonal_count = 0
        for vertexB in unexplored:
            orthogonal_count = 0
            for vertexC in Bunch:
                if vertexC == vertexB:
                    continue
                a, _ = calculate_a_b(matrix, vertices, vertex1, vertexB, vertexC, rate, min_distance, index_of = index_of)
                if a == 0:
                    orthogonal_count += 1
                if a < all_a_min:
                    all_a_min = a
                    a_min_vertex = vertexB
            if orthogonal_count > max_orthogonal_count:
                max_orthogonal_count = orthogonal_count
                Base_vertex = vertexB
        if max_orthogonal_count != 0:
            Bunch_Base.append(Base_vertex)
            unexplored.remove(Base_vertex)
            for vertexC in unexplored.copy():
                a, _ = calculate_a_b(matrix, vertices, vertex1, Base_vertex, vertexC, rate, min_distance, index_of = index_of)
                if a != 0:
                    unexplored.remove(vertexC)
        else:
            Bunch_Base.append(a_min_vertex)
            unexplored = []
    if logger is not None:
        logger(
            "(Base selection) result",
            a=vertex1,
            B=list(Bunch),
            Base=list(Bunch_Base),
        )
    return Bunch_Base

def find_max_distance(vertices, matrix, unexplored_vertices):
    def get_vertex_indices(vertices, target_vertices):
        return [vertices.index(v) for v in target_vertices if v in vertices]
    distance_matrix = np.asarray(matrix, dtype=float)
    unexplored_indices = get_vertex_indices(vertices, unexplored_vertices)
    sub_matrix = distance_matrix[np.ix_(unexplored_indices, unexplored_indices)]
    np.fill_diagonal(sub_matrix, -np.inf)
    max_idx = np.unravel_index(np.argmax(sub_matrix, axis=None), sub_matrix.shape)
    max_distance = sub_matrix[max_idx]
    vertex1, vertex2 = (vertices[unexplored_indices[i]] for i in max_idx)
    return vertex1, vertex2, max_distance

def find_min_distance_from_vertex(vertices, matrix, vertex):
    def get_vertex_index(vertices, vertex):
        return vertices.index(vertex)
    distance_matrix = np.asarray(matrix, dtype=float)
    vertex_index = get_vertex_index(vertices, vertex)
    distances = distance_matrix[vertex_index].copy()
    distances[vertex_index] = np.inf
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    closest_vertex = vertices[min_index]
    return closest_vertex, min_distance

def find_start_vertex(vertices, matrix, unexplored_vertices):
    vertex1, vertex2, _ = find_max_distance(vertices, matrix, unexplored_vertices)
    vertex1_sum = sum_distances_from_vertex(matrix, vertex1, vertices)
    vertex2_sum = sum_distances_from_vertex(matrix, vertex2, vertices)
    return vertex2 if vertex1_sum < vertex2_sum else vertex1

def find_surrounding_vertices(vertices, matrix, explored_vertices, vertex1, rate, min_distance, exclude_explored: bool = False, deadline: float | None = None,):
    index_of = {v: i for i, v in enumerate(vertices)}
    matrix_r = matrix.copy()
    surrounding_candidates = vertices.copy()
    
    # unreV モードでは、周辺点候補から探索済み頂点を除外する
    if exclude_explored:
        for vertex_x in explored_vertices:
            matrix_r, surrounding_candidates = remove_vertex(matrix_r, surrounding_candidates, vertex_x)
    
    surrounding_vertices = []
    while len(surrounding_candidates) > 1:
        _check_timeout(deadline)
        closest_vertex_B, _ = find_min_distance_from_vertex(surrounding_candidates, matrix_r, vertex1)
        vertices_C = surrounding_candidates.copy()
        vertices_C.remove(vertex1)
        vertices_C.remove(closest_vertex_B)
        for vertex_C in vertices_C:
            _, b = calculate_a_b(matrix, vertices, vertex1, closest_vertex_B, vertex_C, rate, min_distance, index_of = index_of)
            # rate*b <= D_AB を満たす頂点を削除する
            if b == 0:
                matrix_r, surrounding_candidates = remove_vertex(matrix_r, surrounding_candidates, vertex_C)    
        surrounding_vertices.append(closest_vertex_B)
        matrix_r, surrounding_candidates = remove_vertex(matrix_r, surrounding_candidates, closest_vertex_B)  
        if len(surrounding_candidates) == 2:
            closest_vertex_B, _ = find_min_distance_from_vertex(surrounding_candidates, matrix_r, vertex1)
            surrounding_vertices.append(closest_vertex_B)
            matrix_r, surrounding_candidates = remove_vertex(matrix_r, surrounding_candidates, closest_vertex_B)
        #print(vertex1)
        #print(surrounding_vertices)
    return [v for v in surrounding_vertices if v not in explored_vertices]

def generate_alpha_indices(n):
    indices = []
    for i in range(n):
        label = ''
        j = i
        while True:
            j, rem = divmod(j, 26)
            label = chr(65 + rem) + label
            if j == 0:
                break
            j -= 1
        indices.append(label)
    return indices

def get_edges_sorted_by_weight(G):
    edges_with_weights = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    return sorted(edges_with_weights, key=lambda x: x[2], reverse=True)

def get_total_graph_length(G: nx.Graph) -> float:
    return sum(data.get("weight", 1.0) for _, _, data in G.edges(data=True))

def get_vertex_index(vertices, vertex):
    return vertices.index(vertex)

def graph_to_distance_matrix(G):
    vertices = list(G.nodes)
    distance_matrix = nx.floyd_warshall_numpy(G, weight='weight')
    return pd.DataFrame(distance_matrix, index=vertices, columns=vertices)

def list_minus(set1, set2):
    return [x for x in set1 if x not in set2]

def _deadline_from_limit(start_time: float, time_limit_sec: float | None) -> float | None:
    return (start_time + time_limit_sec) if (time_limit_sec is not None) else None

def _check_timeout(deadline: float | None):
    if deadline is not None and time.time() > deadline:
        raise TimeoutError("Time limit exceeded for this CSV.")

def main_batch(
    input_dir: Path,
    output_dir: str,
    rate: float,
    rate_XD: float,
    rate_vertex: float,
    rate_edge: float,
    edge_prune_k: int,
    DRAW: str,
    SURROUNDING_MODE: str,
    mds_dim: int,
    time_limit_sec: float | None = None,
):
    csv_files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.dist")))
    for csv_path in csv_files:
        print(f"\n=== {csv_path.name} を処理中 ===")
        try:
            run_for_one_csv(
                file_path=str(csv_path),
                output_dir=output_dir,
                show_plot=False,
                rate=rate,
                rate_XD=rate_XD,
                rate_vertex=rate_vertex,
                rate_edge=rate_edge,
                edge_prune_k=edge_prune_k,
                DRAW=DRAW,
                SURROUNDING_MODE=SURROUNDING_MODE,
                mds_dim=mds_dim,
                time_limit_sec=time_limit_sec,
            )
        except TimeoutError:
            print(f"[TIMEOUT] {csv_path.name} は制限時間超過のためスキップして次へ進みます。")
            continue

def make_unexplored_vertices(vertices, explored_vertices):
    explored = set(explored_vertices)
    return [v for v in vertices if v not in explored]

def process_csv(file_path: str = None):
    """入力ファイルを読み込み、(matrix, vertices, file_path) を返す。

    対応形式:
      - .csv : これまで通り（ヘッダ有無に対応）
      - .dist: SplitsTree の NEXUS 距離形式
    """
    if file_path is None:
        raise ValueError("file_path is required")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".dist":
        matrix, vertices = read_dist_nexus(file_path)
        return matrix, vertices, file_path

    # --- default: csv ---
    data, file_path = readcsv(file_path)
    try:
        float(data[0][0])
        header_present = False
    except ValueError:
        header_present = True
    if not header_present:
        alpha_indices = generate_alpha_indices(len(data[0]))
        data = [[alpha_indices[i]] + data[i] for i in range(len(data))]
        data = [['Index'] + alpha_indices] + data
    df = pd.DataFrame(data[1:], columns=data[0])
    matrix = df.iloc[:, 1:].values.astype(float)
    vertices = df.columns[1:].tolist()
    return matrix, vertices, file_path

def readcsv(file_path: str = None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data, file_path

def remove_redundant_edges(G, edges_sorted, original_vertices, rate_vertex):    
    G_copy = G.copy()    
    # 各辺を重みが大きい順に処理
    for u, v, weight in edges_sorted:
        if G_copy.has_edge(u, v):
            G_copy.remove_edge(u, v)
            # (u, v) 辺が余分かどうかを判定するために、u と v の間に他のパスが存在するか確認
            if nx.has_path(G_copy, u, v):
                length, _ = nx.single_source_dijkstra(G_copy, u, target=v)
                if rate_vertex * length < (1 + rate_vertex) * weight:
                    # (u, v) を迂回するパスがほぼ(u,v)間距離と変わらないなら(u,v)削除
                    G.remove_edge(u, v)
                else:
                    G_copy.add_edge(u, v, weight=weight)
            else:
                # 他のパスが存在しない場合は (u, v) 辺を元に戻す
                G_copy.add_edge(u, v, weight=weight)
    edges_list = list(G.edges(data=True))
    for edge in edges_list:
        vertex1, vertex2, data = edge
        weight = data['weight']
        G_copy = G.copy()
        G_copy.remove_edge(vertex1, vertex2)
        judge = True
        vertices2 = original_vertices.copy()
        for vertexA in original_vertices:
            vertices2.remove(vertexA)
            if vertices2:
                for vertexB in vertices2:
                    if nx.has_path(G_copy, vertexA, vertexB):
                        before_length, _ = nx.single_source_dijkstra(G, vertexA, target=vertexB)
                        after_length, _ = nx.single_source_dijkstra(G_copy, vertexA, target=vertexB)
                        if rate_vertex * after_length > (1 + rate_vertex) * before_length:
                            judge = False
                            break
                    else:
                        judge = False
                        break
            if not judge:
                break
        if judge:
            G.remove_edge(vertex1, vertex2)
        else:
            G_copy.add_edge(vertex1, vertex2, weight=weight)
    return G

def remove_vertex(matrix, vertices, vertex):
    vertex_index = get_vertex_index(vertices, vertex)
    new_matrix = np.delete(matrix, vertex_index, axis=0)
    new_matrix = np.delete(new_matrix, vertex_index, axis=1)
    new_vertices = [v for v in vertices if v != vertex]
    return new_matrix, new_vertices

def remove_vertex_from_G(G, added_vertex, original_vertices, rate_vertex, count):
    vertices1 = original_vertices.copy()
    vertices2 = vertices1.copy()
    judge = True
    G_copy = G.copy()
    G_copy.remove_node(added_vertex)
    for vertex1 in vertices1:
        vertices2.remove(vertex1)
        if vertices2:
            for vertex2 in vertices2:
                before_length, _ = nx.single_source_dijkstra(G, vertex1, target=vertex2)
                if nx.has_path(G_copy, vertex1, vertex2):
                    after_length, _ = nx.single_source_dijkstra(G_copy, vertex1, target=vertex2)
                    if rate_vertex * after_length > (1 + rate_vertex) * before_length:
                        judge = False
                        break
                else:
                    judge = False
                    break
        if not judge:
            break
    if judge:
        print(f"{added_vertex} is removed.")
        G.remove_node(added_vertex)
        count -= 1
    return G, count

def remove_vertices_from_G(G, original_vertices, added_vertices, rate_vertex, count):
    for added_vertex in added_vertices:
        G, count = remove_vertex_from_G(G, added_vertex, original_vertices, rate_vertex, count)
    return G, count

def save_evaluation_summary_row(
    output_dir: str,
    file_path: str,
    status: str,
    rate: float,
    rate_XD: float,
    rate_edge: float,
    added_vertices: int | None,
    total_length: float | None,
    exec_time: float,
    max_ratio: float | None,
    max_pair,
    mae: float | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "実行結果.csv")
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "input_file",
                "status",
                "rate",
                "rate_XD",
                "rate_edge",
                "added_vertices",
                "total_length",
                "exec_time_sec",
                "max_rel_error_abs",
                "max_error_pair_i",
                "max_error_pair_j",
                "mae",
            ])

        def _fmt_opt(x, fmt: str = "{:.17g}"):
            if x is None:
                return ""
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, float):
                return fmt.format(x)
            return str(x)

        # max_pair can be a tuple like (i,j) or None
        mp0 = ""
        mp1 = ""
        if max_pair is not None:
            try:
                mp0 = max_pair[0]
                mp1 = max_pair[1]
            except Exception:
                mp0 = ""
                mp1 = ""

        # --- PATCH: rate_edge column: write percentile if rate_edge_pct > 0 ---
        if edge_prune_k is not None and int(edge_prune_k) > 0:
            rate_edge_repr = f"k:{_fmt_opt(int(edge_prune_k))}"
        else:
            rate_edge_repr = _fmt_opt(float(rate_edge))

        writer.writerow([
            os.path.basename(file_path),
            status,
            _fmt_opt(float(rate)),
            _fmt_opt(float(rate_XD)),
            rate_edge_repr,
            _fmt_opt(added_vertices),
            _fmt_opt(total_length),
            f"{exec_time:.6f}",
            _fmt_opt(max_ratio),
            mp0,
            mp1,
            _fmt_opt(mae),
        ])

def save_network_as_nexus(G, input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, base_name + ".nex") 
    nexus_str = "#NEXUS\n\nBEGIN NETWORK;\n"
    n_vertices = len(G.nodes)
    n_edges = len(G.edges)
    nexus_str += f"DIMENSIONS nVertices={n_vertices} nEdges={n_edges};\n"
    nexus_str += "VERTICES"
    for i, node in enumerate(G.nodes(), start=1):
        nexus_str += f"\n    id={i} label='{node}',"
    nexus_str=nexus_str.rstrip(',')
    nexus_str += "\n;\nEDGES"
    for i, (u, v, data) in enumerate(G.edges(data=True), start=1):
        weight = data.get("weight", 1.0)
        u_id = list(G.nodes).index(u) + 1
        v_id = list(G.nodes).index(v) + 1
        nexus_str += f"\n    id={i} sid={u_id} tid={v_id} weight={weight} w={weight},"
    nexus_str=nexus_str.rstrip(',')
    nexus_str += "\n;\nEND; [NETWORK]\n"
    with open(output_file, "w") as f:
        f.write(nexus_str)
    print(f"NEXUS形式で保存: {output_file}")


# === 新規追加: .dist, .stree6, MDS helper ===


def compute_mds_coords_for_graph(G: nx.Graph, mds_dim: int) -> tuple[list[str], np.ndarray]:
    """実現グラフ距離（最短路距離）からMDS座標を計算する。"""
    dist_mat = nx.floyd_warshall_numpy(G, weight="weight")
    if mds_dim not in (2, 3):
        raise ValueError(f"mds_dim must be 2 or 3, got {mds_dim}")

    mds = MDS(
        n_components=mds_dim,
        dissimilarity='precomputed',
        random_state=0
    )
    coords = mds.fit_transform(dist_mat)

    # --- Axis alignment (PCA) ---
    coords = coords - coords.mean(axis=0, keepdims=True)
    _u, _s, _vt = np.linalg.svd(coords, full_matrices=False)
    coords = coords @ _vt.T

    # Optional: fix sign to make orientation deterministic
    for d in range(coords.shape[1]):
        if coords[:, d].sum() < 0:
            coords[:, d] *= -1

    # --- Extra: rotate so dominant edge direction becomes axis-aligned (2D only) ---
    if mds_dim == 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        idx_of = {node: idx for idx, node in enumerate(nodes)}
        angles: list[float] = []
        for (u, v) in G.edges():
            iu = idx_of[u]
            iv = idx_of[v]
            dx = coords[iv, 0] - coords[iu, 0]
            dy = coords[iv, 1] - coords[iu, 1]
            if dx == 0 and dy == 0:
                continue
            ang = np.arctan2(dy, dx)
            ang = abs(ang) % (np.pi / 2)
            angles.append(float(ang))

        if len(angles) >= 5:
            a = np.array(angles, dtype=float)
            hist, bin_edges = np.histogram(a, bins=180, range=(0.0, np.pi / 2))
            k = int(np.argmax(hist))
            theta = 0.5 * (bin_edges[k] + bin_edges[k + 1])
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, -s], [s, c]], dtype=float)
            coords = coords @ R.T

    return list(G.nodes()), coords




def smooth_graph(graph, count):
    def is_added_vertex_label(value) -> bool:
        # 追加頂点は "m123" のような形式に統一
        if not isinstance(value, str):
            return False
        return value.startswith("m") and value[1:].isdigit()
    smoothed_graph = graph.copy()
    added_edges_by_smoothing: set[tuple[str, str]] = set()
    def norm(u: str, v: str) -> tuple[str, str]:
        return (u, v) if u <= v else (v, u)
    while True:
        nodes_to_remove = [node for node in smoothed_graph.nodes
                           if smoothed_graph.degree[node] == 2 and is_added_vertex_label(node)]
        if not nodes_to_remove:
            break
        acc_sum = {}
        acc_cnt = {}
        for node in nodes_to_remove:
            if node not in smoothed_graph:
                continue
            neighbors = list(smoothed_graph.neighbors(node))
            if len(neighbors) == 2:
                node_a, node_b = neighbors
                weight_a = smoothed_graph[node][node_a]['weight']
                weight_b = smoothed_graph[node][node_b]['weight']
                new_weight = weight_a + weight_b
                key = tuple(sorted((node_a, node_b)))
                acc_sum[key] = acc_sum.get(key, 0.0) + new_weight
                acc_cnt[key] = acc_cnt.get(key, 0) + 1
                smoothed_graph.remove_node(node)
                count -= 1
                added_edges_by_smoothing.add(norm(node_a, node_b))
        for (u, v), s in acc_sum.items():
            c = acc_cnt[(u, v)]
            if smoothed_graph.has_edge(u, v):
                s += smoothed_graph[u][v]['weight']
                c += 1
                smoothed_graph.remove_edge(u, v)
            average_weight = s / c
            smoothed_graph.add_edge(u, v, weight=average_weight)
    return smoothed_graph, count, added_edges_by_smoothing

def sum_distances_from_vertex(matrix, vertex, vertices):
    vertex_index = get_vertex_index(vertices, vertex)
    distances = matrix[vertex_index].copy()
    distances[vertex_index] = 0
    return np.sum(distances)

def XD_calculation(matrix, vertices, vertexA, vertexB, vertexC, vertexD):
    index_A = vertices.index(vertexA)
    index_B = vertices.index(vertexB)
    index_C = vertices.index(vertexC)
    index_D = vertices.index(vertexD)
    D_AB = calculate_distances(matrix, (index_A, index_B))
    D_BC = calculate_distances(matrix, (index_B, index_C))
    D_CA = calculate_distances(matrix, (index_C, index_A))
    D_DA = calculate_distances(matrix, (index_D, index_A))
    D_DB = calculate_distances(matrix, (index_D, index_B))
    D_DC = calculate_distances(matrix, (index_D, index_C))
    D_AX = (D_AB - D_BC + D_CA) / 2
    D_AY = (D_AB - D_DB + D_DA) / 2
    D_AW = (D_CA - D_DC + D_DA) / 2
    D_BX = (D_AB - D_CA + D_BC) / 2
    D_BY = (D_AB + D_DB - D_DA) / 2
    D_BZ = (D_BC - D_DC + D_DB) / 2
    D_CX = (D_BC + D_CA - D_AB) / 2
    D_CZ = (D_BC + D_DC - D_DB) / 2
    D_CW = (D_CA + D_DC - D_DA) / 2
    D_DY = (-D_AB + D_DB + D_DA) / 2
    D_DZ = (-D_BC + D_DC + D_DB) / 2
    D_DW = (-D_CA + D_DC + D_DA) / 2
    a = min(D_AX, D_AY, D_AW)
    b = min(D_BX, D_BY, D_BZ)
    c = min(D_CX, D_CZ, D_CW)
    r1 = D_AX + D_DA - 2 * a
    r2 = D_BX + D_DB - 2 * b
    r3 = D_CX + D_DC - 2 * c
    D_XD = min(r1, r2, r3)
    return D_XD

# 閾値用関数（最小距離）
"""
def find_min_distance(matrix):
    distance_matrix = np.array(matrix, dtype=float)
    np.fill_diagonal(distance_matrix, np.inf)
    return np.min(distance_matrix)
"""
# 閾値用関数（中央値）
def find_min_distance(matrix):
    n = len(matrix)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            d = matrix[i][j]
            if d > 0:
                distances.append(d)
    return float(np.median(distances))

def run_for_one_csv(
    file_path: str,
    output_dir: str,
    show_plot: bool,
    rate: float,
    rate_XD: float,
    rate_vertex: float,
    rate_edge: float,
    edge_prune_k: int,
    DRAW: str,
    SURROUNDING_MODE: str,
    mds_dim: int,
    time_limit_sec: float | None = None,
):
    start_time = time.time()
    deadline = _deadline_from_limit(start_time, time_limit_sec)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    logger, _events, _flush_logs = make_step_logger(output_dir, base_name)

    # --- edge pruning mode selection ---
    try:
        matrix, vertices, file_path = process_csv(file_path)
        original_n = len(vertices)
        logger(
            "Init",
            input_file=os.path.basename(file_path),
            n_original=original_n,
            rate=rate,
            rate_XD=rate_XD,
            rate_vertex=rate_vertex,
            rate_edge=rate_edge,
            edge_prune_k=edge_prune_k,
            SURROUNDING_MODE=SURROUNDING_MODE,
            DRAW=DRAW,
            mds_dim=mds_dim,
            time_limit_sec=time_limit_sec,
        )
        min_distance = find_min_distance(matrix)
        G = nx.Graph()
        count = 0
        original_vertices = vertices.copy()
        explored_vertices = []
        unexplored_vertices = vertices.copy()
        iter_no = 0

        # このループ内でグラフ構造を外側から調べて埋めていく
        while len(unexplored_vertices) >= 2:
            _check_timeout(deadline)
            iter_no += 1

            # snapshot for per-iteration diff
            _V_before = set(vertices)
            _E_before = {tuple(sorted((u, v))) for (u, v) in G.edges()}

            logger(
                "While: V\\R != ∅",
                iter_no=iter_no,
                added_vertex_counter=count,
                explored_size=len(explored_vertices),
                unexplored_size=len(unexplored_vertices),
                explored=list(explored_vertices),
                unexplored=list(unexplored_vertices),
            )
            print("追加頂点数:", count, "未探索頂点数:", len(unexplored_vertices))
            vertex1 = find_start_vertex(vertices, matrix, unexplored_vertices)
            logger("Choose reference a ∈ U", iter_no=iter_no, a=vertex1)
            print("開始点（vertex1）:", vertex1)
            print("探索済頂点:", explored_vertices)

            logger(
                "Define candidate set C(a)",
                a=vertex1,
                selection_type=SURROUNDING_MODE,
                note="unreV excludes R from C(a); allV uses all vertices then filters by R at return",
            )
            # allV or unreV
            exclude_explored = (SURROUNDING_MODE == "unreV")
            surrounding_vertices = find_surrounding_vertices(
                vertices, matrix, explored_vertices, vertex1, rate, min_distance,
                exclude_explored=exclude_explored,
                deadline=deadline,
            )

            if not surrounding_vertices:
                logger("Surrounding result: S(a)=∅ -> skip a", a=vertex1)
                logger(
                    "Iteration summary",
                    iter_no=iter_no,
                    a=vertex1,
                    skipped=True,
                    added_vertices=[],
                    added_edges=[],
                )
                explored_vertices.append(vertex1)
                unexplored_vertices.remove(vertex1)
                continue
            Neighborhood, indirect_vertices = classify_surrounding_vertices(
                vertices, matrix, surrounding_vertices, vertex1, rate, min_distance, logger=logger
            )
            if Neighborhood:
                add_edge_to_Neighborhood(G, vertices, matrix, Neighborhood, vertex1, logger=logger)
            Bunch_list = find_all_Bunches(vertices, matrix, indirect_vertices, vertex1, rate, min_distance, logger=logger)
            for Bunch in Bunch_list:
                Bunch_Base = find_Bunch_Base(vertices, matrix, Bunch, vertex1, rate, min_distance, logger=logger)
                matrix, vertices, count = add_new_vertex(
                    G, vertices, matrix, Bunch, Bunch_Base, vertex1,
                    count, rate, rate_XD, min_distance,
                    deadline=deadline,
                    logger=logger,
                )
                indirect_vertices = list_minus(indirect_vertices, Bunch)
            # per-iteration diff (what was added in this iteration)
            _V_after = set(vertices)
            _E_after = {tuple(sorted((u, v))) for (u, v) in G.edges()}

            _added_vertices = sorted(list(_V_after - _V_before))
            _added_edges_keys = sorted(list(_E_after - _E_before))
            _added_edges = []
            for (u, v) in _added_edges_keys:
                try:
                    w = float(G[u][v].get('weight', 1.0))
                except Exception:
                    w = G[u][v].get('weight', 1.0)
                _added_edges.append({"u": u, "v": v, "weight": w})

            logger(
                "Iteration summary",
                iter_no=iter_no,
                a=vertex1,
                skipped=False,
                added_vertices=_added_vertices,
                added_edges=_added_edges,
            )
            logger("Update R ← R∪{a}", a=vertex1)
            explored_vertices.append(vertex1)
            unexplored_vertices = make_unexplored_vertices(vertices, explored_vertices)

        logger(
            "Main loop finished",
            final_V_size=len(vertices),
            final_R_size=len(explored_vertices),
            G_nodes=G.number_of_nodes(),
            G_edges=G.number_of_edges(),
        )

        # --- 出力直前の簡略化：次数2(追加頂点)のスムージングと、その結果生じた冗長三角形の削除 + 長辺除去 を収束するまで反復 ---
        # --- 出力直前の簡略化（1思想） ---
        # Step1: 先に冗長な長辺を削除（bridgeを毎回再判定）
        # Step2: 次数2(追加頂点)のスムージング（収束まで）
        # Step3: スムージングで生じた冗長三角形の縮約
        # Step2': triangle後に新たに次数2が出るので、もう一度スムージング

        # Step1
        removed_edges = 0
        if (rate_edge is not None) and (float(rate_edge) > 0):
            G, removed_edges = prune_long_edges(G, rate_edge=rate_edge, base_len=min_distance)
            if removed_edges > 0:
                print(f"[post] prune_long_edges(min_distance) removed={removed_edges} thr={float(rate_edge)*float(min_distance):.6g}")
        elif (edge_prune_k is not None) and (int(edge_prune_k) > 0):
            G, removed_edges, thr = prune_long_edges_by_count(G, edge_prune_k=int(edge_prune_k))
            if removed_edges > 0:
                print(f"[post] prune_long_edges_by_count K={int(edge_prune_k)} removed={removed_edges} thr~{(thr if thr is not None else float('nan')):.6g}")
        logger(
            "Postprocess Step1: edge pruning",
            removed_edges=removed_edges,
            G_edges=G.number_of_edges(),
            G_nodes=G.number_of_nodes(),
        )

        # Step2
        G, count, smooth_added_edges = smooth_graph(G, count)
        logger(
            "Postprocess Step2: smoothing",
            G_edges=G.number_of_edges(),
            G_nodes=G.number_of_nodes(),
            smoothed_touch_edges=sorted(list(smooth_added_edges))[:50],
        )

        # Step3
        G, count = collapse_triangles_touching_edges(G, count, smooth_added_edges)
        logger(
            "Postprocess Step3: triangle collapse",
            G_edges=G.number_of_edges(),
            G_nodes=G.number_of_nodes(),
        )

        # Step2'
        G, count, _smooth_added_edges2 = smooth_graph(G, count)
        logger(
            "Postprocess Step2': smoothing (again)",
            G_edges=G.number_of_edges(),
            G_nodes=G.number_of_nodes(),
        )

        # --- 評価指標の計算 ---
        distance_matrix_df = graph_to_distance_matrix(G)
        filtered_distance_matrix_df = filter_alphabetic_indices(distance_matrix_df)
        ext_in = os.path.splitext(file_path)[1].lower()
        if ext_in == ".dist":
            orig_mat, orig_vertices = read_dist_nexus(file_path)
            csv_distance_matrix_df = pd.DataFrame(orig_mat, index=orig_vertices, columns=orig_vertices)
        else:
            csv_data, _ = readcsv(file_path)
            try:
                float(csv_data[0][0])
                header_present = False
            except ValueError:
                header_present = True
            if not header_present:
                alpha_indices = generate_alpha_indices(len(csv_data[0]))
                csv_data = [[alpha_indices[i]] + csv_data[i] for i in range(len(csv_data))]
                csv_data = [['Index'] + alpha_indices] + csv_data
            csv_distance_matrix_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            csv_distance_matrix_df.set_index(csv_distance_matrix_df.columns[0], inplace=True)
            csv_distance_matrix_df = csv_distance_matrix_df.apply(pd.to_numeric, errors="raise")
        filtered_distance_matrix_df = filtered_distance_matrix_df.loc[csv_distance_matrix_df.index, csv_distance_matrix_df.columns]
        result_distance_matrix_df = divide_distance_matrices(
            filtered_distance_matrix_df,
            csv_distance_matrix_df
        )
        print("距離比行列（実現 / 元）：")
        print(result_distance_matrix_df)

        # 相対誤差行列 (d_real - d_orig) / d_orig = (実現/元) - 1
        rel_error_df = result_distance_matrix_df - 1.0
        rel_error_df = rel_error_df.mask(
            np.eye(rel_error_df.shape[0], dtype=bool)
        )
        # 絶対値を取った最大相対誤差
        abs_rel_error_df = rel_error_df.abs()
        max_abs_rel_error = abs_rel_error_df.stack().max()
        max_abs_loc = abs_rel_error_df.stack().idxmax()
        max_over_error = rel_error_df.stack().max()   # 最も長くなった
        max_over_loc = rel_error_df.stack().idxmax()
        max_under_error = rel_error_df.stack().min()  # 最も短くなった
        max_under_loc = rel_error_df.stack().idxmin()
        print(
            f"最大相対誤差（絶対値） |Δd|/d_orig = {max_abs_rel_error:.6g} "
            f"at [{max_abs_loc[0]}][{max_abs_loc[1]}]"
        )
        print(
            f"最大過大評価 (d_real - d_orig)/d_orig = {max_over_error:.6g} "
            f"at [{max_over_loc[0]}][{max_over_loc[1]}]"
        )
        print(
            f"最大過小評価 (d_real - d_orig)/d_orig = {max_under_error:.6g} "
            f"at [{max_under_loc[0]}][{max_under_loc[1]}]"
        )

        total_length = get_total_graph_length(G)
        print(f"グラフ全長: {total_length}")
        mae = compute_mae(filtered_distance_matrix_df, csv_distance_matrix_df)
        print(f"平均絶対誤差 (MAE): {mae}")

        # added_vertices は count ではなく、完成グラフの頂点数 - 元の頂点数
        added_vertices_for_csv = int(len(G.nodes) - original_n)

        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(file_path))[0]}_{DRAW}.pdf"
        )

        # グラフの描画
        if DRAW == 'normal':
            pos = nx.kamada_kawai_layout(G)
            labels = {node: node for node in G.nodes if isinstance(node, str)}
            plt.figure(figsize=(8, 6))
            nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=30, font_size=5, width=0.5)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
            plt.title('Graph with Weighted Edges')
            plt.tight_layout()
            plt.savefig(pdf_path)
            if show_plot:
                plt.show()
            plt.close()
        elif DRAW == "MDS":
            nodes, coords = compute_mds_coords_for_graph(G, mds_dim)
            idx_of = {node: idx for idx, node in enumerate(nodes)}
            if mds_dim == 2:
                plt.figure(figsize=(10, 10))
                for idx, node in enumerate(nodes):
                    plt.scatter(coords[idx, 0], coords[idx, 1], s=10, color='skyblue')
                    plt.text(coords[idx, 0], coords[idx, 1], node, fontsize=6)

                for (u, v, data) in G.edges(data=True):
                    i = idx_of[u]
                    j = idx_of[v]
                    plt.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        linewidth=0.5, color='gray'
                    )

                ax = plt.gca()
                ax.set_axis_off()
                ax.set_aspect('equal', adjustable='box')

                # --- avoid cropping: set explicit limits with padding ---
                xs = coords[:, 0]
                ys = coords[:, 1]
                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                xr = max(1e-12, x_max - x_min)
                yr = max(1e-12, y_max - y_min)
                pad = 0.06 * max(xr, yr)
                ax.set_xlim(x_min - pad, x_max + pad)
                ax.set_ylim(y_min - pad, y_max + pad)

                plt.tight_layout(pad=0)
                # pad_inches を少し入れて、bbox_inches='tight' でも切れにくくする
                plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15)
                if show_plot:
                    plt.show()
                plt.close()

            else:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                for idx, node in enumerate(nodes):
                    ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=10, color='skyblue')
                    ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2], node, fontsize=6)

                for (u, v, data) in G.edges(data=True):
                    i = idx_of[u]
                    j = idx_of[v]
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        linewidth=0.5, color='gray'
                    )

                # ---- figure styling: no axes/title, big graph ----
                ax.set_axis_off()

                # --- avoid cropping: set explicit limits with padding ---
                xs = coords[:, 0]
                ys = coords[:, 1]
                zs = coords[:, 2]
                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                z_min, z_max = float(zs.min()), float(zs.max())
                xr = max(1e-12, x_max - x_min)
                yr = max(1e-12, y_max - y_min)
                zr = max(1e-12, z_max - z_min)
                pad = 0.06 * max(xr, yr, zr)
                ax.set_xlim(x_min - pad, x_max + pad)
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.set_zlim(z_min - pad, z_max + pad)

                plt.tight_layout(pad=0)
                plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.15)
                if show_plot:
                    plt.show()
                plt.close(fig)

        exec_time = time.time() - start_time
        print(f"処理時間: {exec_time:.2f} 秒")

        save_network_as_nexus(G, file_path, output_dir)

        save_evaluation_summary_row(
            output_dir=output_dir,
            file_path=file_path,
            status="OK",
            rate=rate,
            rate_XD=rate_XD,
            rate_edge=rate_edge,
            added_vertices=added_vertices_for_csv,
            total_length=total_length,
            exec_time=exec_time,
            max_ratio=max_abs_rel_error,
            max_pair=max_abs_loc,
            mae=mae,
        )
        _flush_logs()

    except TimeoutError:
        exec_time = time.time() - start_time
        print(f"[TIMEOUT] {os.path.basename(file_path)} は制限時間超過のため中断しました。")

        # TIMEOUT でも1行書く（計測できない値は空欄）
        save_evaluation_summary_row(
            output_dir=output_dir,
            file_path=file_path,
            status="TIMEOUT",
            rate=rate,
            rate_XD=rate_XD,
            rate_edge=rate_edge,
            added_vertices=None,
            total_length=None,
            exec_time=exec_time,
            max_ratio=None,
            max_pair=None,
            mae=None,
        )
        try:
            _flush_logs()
        except Exception:
            pass
        return

if __name__ == '__main__':
    # 周辺点探索モード
    #   "allV"  : 全頂点を対象に周辺点探索
    #   "unreV" : 未探索頂点のみを対象に周辺点探索
    SURROUNDING_MODE = "allV"   # or "unreV"

    # 近似される割合を制御する変数（逆数）
    rate = rate_XD = 200
    
    rate_vertex = 0

    # 長すぎる辺の削除しきい値: weight > rate_edge * min_distance を削除（bridgeは削除しない）
    rate_edge =  0.0
    edge_prune_k = 0  # 上位K本の長い辺を削除（rate_edge と排他的）

    # グラフ描画モードの切り替え（'normal'なら通常描画、'MDS'ならMDS描画）
    DRAW = 'normal'
    MDS_DIM = 2  # 2 or 3

    # 1ファイルあたりの時間制限（秒）。None にすると無制限
    TIME_LIMIT_SEC = 10 * 60   # 例: 2時間

    # 1つのファイルを実行する場合の場合の入出力フォルダ
    DEFAULT_CSV_PATH = '/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_data/others/example1.csv'
    OUTPUT_DIR_SINGLE = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm"

    # 複数のファイルを一括処理する場合の入出力フォルダ
    INPUT_DIR = Path("/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_data/k, n=50/k=640")
    OUTPUT_DIR_BATCH = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/experiment/提案手法1_unreV/artificial_data✅/k, n=50✅/k=640"

    # "single" なら1ファイルだけ、"batch" ならフォルダ内の .csv を全部処理
    MODE = "single"
    if MODE == "single":
        run_for_one_csv(
            file_path=DEFAULT_CSV_PATH,
            output_dir=OUTPUT_DIR_SINGLE,
            show_plot=False,
            rate=rate,
            rate_XD=rate_XD,
            rate_vertex=rate_vertex,
            rate_edge=rate_edge,
            edge_prune_k=edge_prune_k,
            DRAW=DRAW,
            SURROUNDING_MODE=SURROUNDING_MODE,
            mds_dim=MDS_DIM,
            time_limit_sec=TIME_LIMIT_SEC,
        )
    elif MODE == "batch":
        main_batch(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR_BATCH,
            rate=rate,
            rate_XD=rate_XD,
            rate_vertex=rate_vertex,
            rate_edge=rate_edge,
            edge_prune_k=edge_prune_k,
            DRAW=DRAW,
            SURROUNDING_MODE=SURROUNDING_MODE,
            mds_dim=MDS_DIM,
            time_limit_sec=TIME_LIMIT_SEC,
        )