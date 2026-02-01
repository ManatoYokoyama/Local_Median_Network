import csv
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
mpl.use("TkAgg")
mpl.rcParams["figure.dpi"] = 150

# --- Helper functions for time limit ---
def _deadline_from_limit(start_time: float, time_limit_sec: float | None) -> float | None:
    return (start_time + time_limit_sec) if (time_limit_sec is not None) else None

def _check_timeout(deadline: float | None):
    if deadline is not None and time.time() > deadline:
        raise TimeoutError("Time limit exceeded for this CSV.")


# ===============================================================
#  入出力ユーティリティ
# ===============================================================

def readcsv(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                data.append(row)
    return data, file_path

def generate_alpha_indices(n: int):
    indices = []
    for i in range(n):
        label = ""
        j = i
        while True:
            j, rem = divmod(j, 26)
            label = chr(65 + rem) + label
            if j == 0:
                break
            j -= 1
        indices.append(label)
    return indices

def process_csv(file_path: str):
    data, file_path = readcsv(file_path)
    try:
        float(data[0][0])
        header_present = False
    except ValueError:
        header_present = True
    if not header_present:
        alpha_indices = generate_alpha_indices(len(data[0]))
        data = [[alpha_indices[i]] + data[i] for i in range(len(data))]
        data = [["Index"] + alpha_indices] + data
    df = pd.DataFrame(data[1:], columns=data[0])
    D = df.iloc[:, 1:].values.astype(float)
    labels = df.columns[1:].tolist()
    n = D.shape[0]
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            v = 0.5 * (D[i, j] + D[j, i])
            D[i, j] = D[j, i] = v
    return D, labels, file_path

# ===============================================================
#  三点関係 & median 判定
# ===============================================================

def three_point_coeffs(D: np.ndarray, i: int, j: int, k: int):
    dij = D[i, j]
    dik = D[i, k]
    djk = D[j, k]
    a = 0.5 * (dij + dik - djk)
    b = 0.5 * (dij + djk - dik)
    c = 0.5 * (dik + djk - dij)
    a = max(a, 0.0)
    b = max(b, 0.0)
    c = max(c, 0.0)
    return a, b, c

def has_median(D: np.ndarray, i: int, j: int, k: int, eps: float = 1e-10) -> bool:
    """
    三点 {i,j,k} が既存頂点の中に median を持つか
    """
    n = D.shape[0]
    dij = D[i, j]
    dik = D[i, k]
    djk = D[j, k]
    for v in range(n):
        if v in (i, j, k):
            continue
        if (abs(D[i, v] + D[v, j] - dij) < eps and
            abs(D[i, v] + D[v, k] - dik) < eps and
            abs(D[j, v] + D[v, k] - djk) < eps):
            return True
    return False

# ===============================================================
#  式 (1.2) に対応する「median の距離ベクトル」の固定点計算
# ===============================================================

def compute_m_vector_fixed_point(
    D: np.ndarray,
    i: int, j: int, k: int,
    a: float, b: float, c: float,
    max_iter: int,
    eps: float
) -> np.ndarray:
    """
    新頂点 v の「既存頂点 u への距離」ベクトル m(u,v) を
    Karzanov の (1.2)

        m(u,v) = max_{x in V} | m(u,x) - m(x,v) |

    の固定点として近似的に求める。
    """
    n = D.shape[0]
    # v を含めたサイズ n+1 のベクトル（最後の成分 = v 自身）
    m_vec = np.zeros(n + 1, dtype=float)

    # i,j,k への距離は固定（境界条件）
    m_vec[i] = a
    m_vec[j] = b
    m_vec[k] = c
    m_vec[n] = 0.0  # v 自身

    idxs = [i, j, k]
    vals = [a, b, c]

    # 初期値：3点との距離から
    for u in range(n):
        if u in (i, j, k):
            continue
        m0_candidates = []
        for t_idx, s in enumerate(idxs):
            m0_candidates.append(abs(D[u, s] - vals[t_idx]))
        m_vec[u] = max(m0_candidates)

    # 固定点反復
    for _ in range(max_iter):
        m_old = m_vec.copy()
        for u in range(n):
            if u in (i, j, k):
                continue
            max_val = 0.0
            for x in range(n):
                val = abs(D[u, x] - m_old[x])
                if val > max_val:
                    max_val = val
            # x = v
            if abs(m_old[u]) > max_val:
                max_val = abs(m_old[u])
            m_vec[u] = max_val

        m_vec[i] = a
        m_vec[j] = b
        m_vec[k] = c
        m_vec[n] = 0.0

        if np.max(np.abs(m_vec - m_old)) < eps:
            break

    return m_vec

def add_median_vertex_strict(
    D: np.ndarray,
    labels: list[str],
    i: int, j: int, k: int,
    base_name: str,
    fixpoint_max_iter: int,
    fixpoint_eps: float
) -> tuple[np.ndarray, list[str]]:
    """
    三点 {i,j,k} に median 頂点 v を 1 つ追加する。
    - (1.1) で a,b,c を決め
    - (1.2) に対応した固定点計算で m(u,v) を決める
    """
    n = D.shape[0]
    a, b, c = three_point_coeffs(D, i, j, k)

    new_label = f"{base_name}{len(labels) + 1}"
    labels_new = labels + [new_label]

    D_new = np.zeros((n + 1, n + 1), dtype=float)
    D_new[:n, :n] = D

    m_vec = compute_m_vector_fixed_point(
        D, i, j, k, a, b, c,
        max_iter=fixpoint_max_iter,
        eps=fixpoint_eps
    )
    v_idx = n

    for u in range(n):
        D_new[u, v_idx] = D_new[v_idx, u] = m_vec[u]
    D_new[v_idx, v_idx] = 0.0

    return D_new, labels_new

# ===============================================================
#  距離 0 の頂点を同一視して縮約
# ===============================================================

def collapse_zero_distance_vertices_with_map(
    D: np.ndarray,
    labels: list[str],
    original_n: int,
    zero_tol: float,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    d(i,j) が zero_tol 未満の頂点同士を同一視して 1 つの頂点に縮約する。
    ただし、
      - 元頂点 (index < original_n) 同士は絶対に union しない
      - 元頂点と追加頂点が 0（に近い）距離なら union するが、
        代表ラベルは「元頂点側」を優先する。
    戻り値:
      D_new, labels_new, old_to_new_index
    """
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    # zero_tol 未満のペアを union
    for i in range(n):
        for j in range(i + 1, n):
            if abs(D[i, j]) < zero_tol:
                # 両方とも元頂点の場合は同一視しない
                if i < original_n and j < original_n:
                    continue
                union(i, j)

    # root → メンバー集合
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    # root → rep, rep のリスト
    root_to_rep: dict[int, int] = {}
    reps: list[int] = []
    for root, members in groups.items():
        orig_members = [idx for idx in members if idx < original_n]
        if orig_members:
            rep = min(orig_members)
        else:
            rep = min(members)
        root_to_rep[root] = rep
        reps.append(rep)

    # rep をソートして新しいインデックスを決める
    reps_sorted = sorted(reps)
    rep_to_new = {rep: idx for idx, rep in enumerate(reps_sorted)}
    new_labels = [labels[rep] for rep in reps_sorted]

    m = len(reps_sorted)
    D_new = np.zeros((m, m), dtype=float)
    for a, ra in enumerate(reps_sorted):
        for b, rb in enumerate(reps_sorted):
            D_new[a, b] = D[ra, rb]

    # old index -> new index
    old_to_new: list[int] = []
    for i in range(n):
        root = find(i)
        rep = root_to_rep[root]
        new_idx = rep_to_new[rep]
        old_to_new.append(new_idx)

    return D_new, new_labels, old_to_new

# ===============================================================
#  modular closure
#   - triple は常に「元の頂点 X」のみ
#   - 途中で定期的に collapse_zero_distance_vertices
# ===============================================================

def modular_closure_v4_1(
    D: np.ndarray,
    labels: list[str],
    original_n: int,
    collapse_every: int,
    eps_median: float,
    zero_tol: float,
    median_deficit_threshold: float,
    fixpoint_max_iter: int,
    fixpoint_eps: float,
    deadline: float | None = None,
) -> tuple[np.ndarray, list[str], list[int], int]:
    """
    modular closure
    - 三点 {oi,oj,ok} は「元頂点」からのみ選ぶ（oi,oj,ok < original_n）
    - その三点集合は一度だけ combinations で列挙し、以後は固定
    - 各三点について median は高々 1 回だけ追加する（processed_triples）
    - 三点の“歪み”が median_deficit_threshold 以下なら「ノイズ」とみなし拡張しない
    - collapse で 0（に近い）距離頂点を縮約する（元頂点は潰さない）
    """
    D_ext = D.copy()
    labels_ext = list(labels)

    # 元頂点 oi の「現在の行列上の index」
    orig_to_curr = list(range(original_n))

    added_total = 0

    # 三点探索は常に「元頂点のみ」に対して行う（本手法の定義）
    original_triples = list(combinations(range(original_n), 3))

    # すでに処理した三点の集合
    processed_triples: set[tuple[int, int, int]] = set()

    def triple_deviation(ci: int, cj: int, ck: int) -> float:
        """三点 (ci,cj,ck) の歪みの大きさを返す（小さいほど“ほぼ median”）。"""
        dij = D_ext[ci, cj]
        djk = D_ext[cj, ck]
        dik = D_ext[ci, ck]
        d1 = abs(dij + djk - dik)
        d2 = abs(dij + dik - djk)
        d3 = abs(dik + djk - dij)
        return min(d1, d2, d3)

    while True:
        _check_timeout(deadline)
        found = False

        for (oi, oj, ok) in original_triples:
            key = (oi, oj, ok)
            if key in processed_triples:
                continue  # この三点は既に一度処理済み

            ci = orig_to_curr[oi]
            cj = orig_to_curr[oj]
            ck = orig_to_curr[ok]

            # collapse の結果、同じ頂点になっていたら何もしないで「処理済み」
            if len({ci, cj, ck}) < 3:
                processed_triples.add(key)
                continue

            # すでに median を持つなら拡張不要 → 処理済み
            if has_median(D_ext, ci, cj, ck, eps=eps_median):
                processed_triples.add(key)
                continue

            # 三点の歪みがノイズレベル以下なら「拡張不要」とみなしてスキップ
            dev = triple_deviation(ci, cj, ck)
            if dev <= median_deficit_threshold:
                processed_triples.add(key)
                continue

            # ここで初めて「ノイズ以上に歪んでいる medianless triple」と判定されたので、
            # 一度だけ median 頂点を追加する
            D_ext, labels_ext = add_median_vertex_strict(
                D_ext, labels_ext, ci, cj, ck,
                base_name="m",
                fixpoint_max_iter=fixpoint_max_iter,
                fixpoint_eps=fixpoint_eps
            )
            added_total += 1

            # この三点はもう二度と median を追加しない
            processed_triples.add(key)

            found = True
            break  # 1 つ追加したらループを抜けて、また最初から三点探索へ

        if not found:
            # すべての三点について「既に処理済み」になった
            break

        # 一定回数ごとに 0（に近い）距離縮約
        if collapse_every > 0 and (added_total % collapse_every == 0):
            D_ext, labels_ext, old_to_new = collapse_zero_distance_vertices_with_map(
                D_ext, labels_ext, original_n, zero_tol=zero_tol
            )
            # 元頂点のインデックス対応を更新
            orig_to_curr = [old_to_new[idx] for idx in orig_to_curr]

    # 最後に一度まとめて 0（に近い）距離縮約
    D_ext, labels_ext, old_to_new_final = collapse_zero_distance_vertices_with_map(
        D_ext, labels_ext, original_n, zero_tol=zero_tol
    )
    orig_to_curr = [old_to_new_final[idx] for idx in orig_to_curr]
    return D_ext, labels_ext, orig_to_curr, added_total

# ===============================================================
#  距離保存 minimal graph（完全グラフから距離を保ったまま辺削除）
# ===============================================================

def build_distance_preserving_graph(
    D: np.ndarray,
    labels: list[str],
    tol: float,
    deadline: float | None = None,
) -> nx.Graph:
    """
    距離行列 D （labels に対応）を実現する「距離保存かつできるだけ sparsity の高いグラフ」を構成する。

    方針:
      - 辺 (i,j) が「第三頂点 k を経由して距離が再現できる」なら不要とみなし、
        どの k に対しても再現できないときだけ edge として採用する。
      - 具体的には、全ての i<j について、
            ∀k≠i,j について D[i,k] + D[k,j] > D[i,j] + tol
        が成り立つときにのみ (i,j) を edge に採用し、重みを D[i,j] にする。

    計算量は O(n^3)（三重ループ）で、これまでの「完全グラフから
    辺を1本ずつ削除し、そのたびに最短路を再計算する」O(n^5) 実装より
    圧倒的に高速。
    """
    n = len(labels)
    G = nx.Graph()
    G.add_nodes_from(labels)

    # インデックスとラベルの対応を明示（D は labels の順と対応している前提）
    # idx -> label は labels[idx]
    # label -> idx が必要なら下で使う
    # label_to_idx = {lab: i for i, lab in enumerate(labels)}  # 今回は使わない

    for i in range(n):
        _check_timeout(deadline)
        for j in range(i + 1, n):
            dij = float(D[i, j])

            # (i, j) が「第三頂点 k を経由して距離を実現できる」なら edge 不要
            needed = True
            for k in range(n):
                if k == i or k == j:
                    continue
                dik = float(D[i, k])
                dkj = float(D[k, j])

                # k 経由で十分近い距離が実現できるなら (i,j) は不要とみなす
                if dik + dkj <= dij + tol:
                    needed = False
                    break

            if needed:
                u = labels[i]
                v = labels[j]
                G.add_edge(u, v, weight=dij)

    return G

# ===============================================================
#  グラフ構築後：追加頂点どうしを近さで縮約（辺削除後に実施）
# ===============================================================

def collapse_close_added_vertices_in_graph(
    G: nx.Graph,
    original_labels: list[str],
    close_tol: float,
) -> tuple[nx.Graph, dict[str, str]]:
    """
    build_distance_preserving_graph 後のグラフ G に対して、
    追加頂点（original_labels に含まれない頂点）どうしの最短路距離が close_tol 未満なら同一視して縮約する。

    - 元頂点は絶対に縮約しない
    - 元頂点と追加頂点も縮約しない

    戻り値:
      G_new: 縮約後のグラフ
      old_to_new: 旧ラベル -> 新ラベル（代表）
    """
    if close_tol <= 0.0:
        return G, {n: n for n in G.nodes}

    orig_set = set(original_labels)
    nodes = list(G.nodes)
    added_nodes = [n for n in nodes if n not in orig_set]

    # Union-Find for added nodes only
    parent: dict[str, str] = {n: n for n in added_nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic: keep lexicographically smaller as root
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # For each added node, run Dijkstra with cutoff to find close added nodes
    for s, dist_dict in nx.all_pairs_dijkstra_path_length(G, cutoff=close_tol, weight="weight"):
        if s not in parent:
            continue  # only start from added nodes
        for t, d in dist_dict.items():
            if t == s:
                continue
            if t not in parent:
                continue  # only merge added-added
            if d < close_tol:
                union(s, t)

    # Build groups
    groups: dict[str, list[str]] = {}
    for n in added_nodes:
        r = find(n)
        groups.setdefault(r, []).append(n)

    # Representative for each group: lexicographically smallest label
    rep_of: dict[str, str] = {}
    for r, members in groups.items():
        rep = min(members)
        for m in members:
            rep_of[m] = rep

    # old -> new mapping for all nodes
    old_to_new: dict[str, str] = {}
    for n in nodes:
        if n in orig_set:
            old_to_new[n] = n
        else:
            old_to_new[n] = rep_of.get(n, n)

    # Build contracted graph (merge parallel edges by keeping the minimum weight)
    G_new = nx.Graph()
    for n in nodes:
        G_new.add_node(old_to_new[n])

    for u, v, data in G.edges(data=True):
        uu = old_to_new[u]
        vv = old_to_new[v]
        if uu == vv:
            continue
        w = float(data.get("weight", 1.0))
        if G_new.has_edge(uu, vv):
            if w < float(G_new[uu][vv].get("weight", w)):
                G_new[uu][vv]["weight"] = w
        else:
            G_new.add_edge(uu, vv, weight=w)

    return G_new, old_to_new

# ===============================================================
#  評価用関数
# ===============================================================

def get_total_graph_length(G: nx.Graph) -> float:
    return sum(data.get("weight", 1.0) for _, _, data in G.edges(data=True))

def graph_to_distance_matrix(G: nx.Graph) -> pd.DataFrame:
    vertices = list(G.nodes)
    distance_matrix = nx.floyd_warshall_numpy(G, weight="weight")
    return pd.DataFrame(distance_matrix, index=vertices, columns=vertices)

def filter_alphabetic_indices(distance_matrix_df: pd.DataFrame) -> pd.DataFrame:
    idx = distance_matrix_df.index.astype(str)
    cols = distance_matrix_df.columns.astype(str)

    if idx.str.match(r"^[A-Za-z]+$").all() and cols.str.match(r"^[A-Za-z]+$").all():
        return distance_matrix_df.loc[
            idx.str.match(r"^[A-Za-z]+$"),
            cols.str.match(r"^[A-Za-z]+$")
        ]
    else:
        return distance_matrix_df

def compute_mae(filtered_distance_matrix_df: pd.DataFrame,
                csv_distance_matrix_df: pd.DataFrame) -> float:
    error_df = (filtered_distance_matrix_df - csv_distance_matrix_df).abs()
    non_diag_error_df = error_df.mask(np.eye(error_df.shape[0], dtype=bool))
    mae = non_diag_error_df.stack().mean()
    return float(mae)

def divide_distance_matrices(filtered_df: pd.DataFrame,
                             csv_df: pd.DataFrame) -> pd.DataFrame:
    # assert filtered_df.shape == csv_df.shape
    result = filtered_df.divide(csv_df).replace([np.inf, -np.inf], np.nan)
    np.fill_diagonal(result.values, np.diag(filtered_df.values))
    return result

def save_network_as_nexus(G: nx.Graph, input_file: str, output_dir: str):
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
    nexus_str = nexus_str.rstrip(',')

    nexus_str += ";\nEDGES"
    node_list = list(G.nodes)
    for i, (u, v, data) in enumerate(G.edges(data=True), start=1):
        weight = data.get("weight", 1.0)
        u_id = node_list.index(u) + 1
        v_id = node_list.index(v) + 1
        nexus_str += f"\n    id={i} sid={u_id} tid={v_id} weight={weight},"
    nexus_str = nexus_str.rstrip(',')

    nexus_str += "\n;\nEND; [NETWORK]\n"

    with open(output_file, "w") as f:
        f.write(nexus_str)
    print(f"NEXUS ファイルを保存しました: {output_file}")

def save_evaluation_summary_row(
    output_dir: str,
    file_path: str,
    status: str,
    # --- outputs ---
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

        writer.writerow([
            os.path.basename(file_path),
            status,
            _fmt_opt(added_vertices),
            _fmt_opt(total_length),
            f"{exec_time:.6f}",
            _fmt_opt(max_ratio),
            max_pair[0] if max_pair else "",
            max_pair[1] if max_pair else "",
            _fmt_opt(mae),
        ])

# ===============================================================
#  メイン：1 ファイル / バッチ
# ===============================================================

def run_for_one_csv(
    file_path: str,
    output_dir: str,
    show_plot: bool,
    DRAW: str,
    mds_dim: int,
    time_limit_sec: float | None = None,
):
    start_time = time.time()
    deadline = _deadline_from_limit(start_time, time_limit_sec)
    try:
        D_orig, labels_orig, file_path = process_csv(file_path)
        original_n = len(labels_orig)

        print(f"入力ファイル: {file_path}")
        print(f"元の頂点数: {original_n}")

        # ===== strict / theory-aligned (no heuristics) =====
        # Median existence check tolerance (floating arithmetic safeguard)
        eps_median = 1e-12
        # Distance-preserving edge test tolerance
        tol_dist = 1e-12
        # Only collapse exact zero-distance vertices
        zero_tol = 0.0
        # Do not collapse added vertices in the final graph
        graph_collapse_tol = 0.0
        # Extend all medianless triples
        median_deficit_threshold = 0.0

        print(
            f"eps_median = {eps_median:.4g}, tol_dist = {tol_dist:.4g}, "
            f"zero_tol = {zero_tol:.4g}, "
            f"graph_collapse_tol = {graph_collapse_tol:.4g}"
        )
        print(f"median_deficit_threshold = {median_deficit_threshold:.4g}")

        _check_timeout(deadline)
        D_ext, labels_ext, _, added_total = modular_closure_v4_1(
            D_orig,
            labels_orig,
            original_n,
            collapse_every=0,
            eps_median=eps_median,
            zero_tol=zero_tol,
            median_deficit_threshold=median_deficit_threshold,
            fixpoint_max_iter=200,
            fixpoint_eps=1e-7,
            deadline=deadline,
        )

        print(f"modular closure 後の頂点数: {len(labels_ext)} (追加回数: {added_total})")

        added_vertices = len(labels_ext) - original_n
        print(f"元頂点からの純追加頂点数: {added_vertices}")

        G = build_distance_preserving_graph(D_ext, labels_ext, tol=tol_dist, deadline=deadline)

        # 辺削除後のグラフに対して、追加頂点どうしのみ近さで縮約（元頂点は絶対に潰さない）
        G, _old_to_new_graph = collapse_close_added_vertices_in_graph(
            G,
            original_labels=labels_orig,
            close_tol=graph_collapse_tol,
        )

        print(f"[post-graph collapse] グラフ頂点数: {G.number_of_nodes()} / 辺数: {G.number_of_edges()}")
        added_vertices_graph = G.number_of_nodes() - original_n
        print(f"[post-graph collapse] 純追加頂点数: {added_vertices_graph}")

        # 評価用距離行列
        distance_matrix_df = graph_to_distance_matrix(G)
        filtered_distance_matrix_df = filter_alphabetic_indices(distance_matrix_df)

        # 元距離行列の DataFrame（再読込せず process_csv の結果を使う）
        csv_distance_matrix_df = pd.DataFrame(D_orig, index=labels_orig, columns=labels_orig)
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

        # 図と PDF 保存
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(file_path))[0]}_{DRAW}.pdf"
        )

        if DRAW == "normal":
            pos = nx.kamada_kawai_layout(G)
            #pos = nx.spring_layout(G, seed=0)
            plt.figure(figsize=(8, 6))
            nx.draw(
                G, pos,
                with_labels=True,
                node_color="skyblue",
                node_size=50,
                font_size=6,
                width=0.5
            )
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
            plt.title("Graph with Weighted Edges")
            plt.tight_layout()
            plt.savefig(pdf_path)
            if show_plot:
                plt.show()
            plt.close()

        elif DRAW == "MDS":
            from sklearn.manifold import MDS

            dist_mat = nx.floyd_warshall_numpy(G, weight="weight")
            if mds_dim not in (2, 3):
                raise ValueError(f"mds_dim must be 2 or 3, got {mds_dim}")

            mds = MDS(
                n_components=mds_dim,
                dissimilarity='precomputed',
                random_state=0
            )
            coords = mds.fit_transform(dist_mat)

            nodes = list(G.nodes())
            idx_of = {node: idx for idx, node in enumerate(nodes)}

            # --- Axis alignment (PCA) ---
            # MDS coordinates are only determined up to rigid transforms.
            # Align principal axes to (x,y[,z]) so the plot doesn't look arbitrarily rotated.
            coords = coords - coords.mean(axis=0, keepdims=True)
            _u, _s, _vt = np.linalg.svd(coords, full_matrices=False)
            coords = coords @ _vt.T

            # Optional: fix sign to make orientation deterministic
            for d in range(coords.shape[1]):
                if coords[:, d].sum() < 0:
                    coords[:, d] *= -1

            # --- Extra: rotate so dominant edge direction becomes axis-aligned (2D only) ---
            # PCA can still leave a Manhattan-like grid tilted (often suggests ~45°).
            # Estimate a dominant edge direction from edges and rotate it to 0°.
            if mds_dim == 2 and G.number_of_edges() > 0:
                angles: list[float] = []
                for (u, v) in G.edges():
                    iu = idx_of[u]
                    iv = idx_of[v]
                    dx = coords[iv, 0] - coords[iu, 0]
                    dy = coords[iv, 1] - coords[iu, 1]
                    if dx == 0 and dy == 0:
                        continue
                    ang = np.arctan2(dy, dx)
                    # directionless + modulo 90deg (pi/2)
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

            if mds_dim == 2:
                plt.figure(figsize=(8, 6))
                for idx, node in enumerate(nodes):
                    plt.scatter(coords[idx, 0], coords[idx, 1], s=10, color='skyblue')
                    plt.text(coords[idx, 0], coords[idx, 1], node, fontsize=6)

                # edges
                for (u, v, data) in G.edges(data=True):
                    i = idx_of[u]
                    j = idx_of[v]
                    plt.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        linewidth=0.5, color='gray'
                    )

                plt.title("Graph with Weighted Edges (MDS)")
                plt.tight_layout()
                plt.savefig(pdf_path)
                if show_plot:
                    plt.show()
                plt.close()

            else:  # mds_dim == 3
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                for idx, node in enumerate(nodes):
                    ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=10, color='skyblue')
                    ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2], node, fontsize=6)

                # edges
                for (u, v, data) in G.edges(data=True):
                    i = idx_of[u]
                    j = idx_of[v]
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        linewidth=0.5, color='gray'
                    )

                ax.set_title("Graph with Weighted Edges (MDS)")
                plt.tight_layout()
                plt.savefig(pdf_path)
                if show_plot:
                    plt.show()
                plt.close(fig)

        # Print timing after drawing/plotting
        exec_time = time.time() - start_time
        print(f"処理時間: {exec_time:.2f} 秒")

        print(f"PDF 保存: {pdf_path}")

        # NEXUS
        _check_timeout(deadline)
        save_network_as_nexus(G, file_path, output_dir)

        # 実行結果.csv に追記
        save_evaluation_summary_row(
            output_dir=output_dir,
            file_path=file_path,
            status="OK",
            added_vertices=added_vertices_graph,
            total_length=total_length,
            exec_time=exec_time,
            max_ratio=max_abs_rel_error,
            max_pair=max_abs_loc,
            mae=mae,
        )
    except TimeoutError:
        exec_time = time.time() - start_time
        print(f"[TIMEOUT] {os.path.basename(file_path)} は制限時間超過のためスキップします。")
        os.makedirs(output_dir, exist_ok=True)
        save_evaluation_summary_row(
            output_dir=output_dir,
            file_path=file_path,
            status="TIMEOUT",
            added_vertices=None,
            total_length=None,
            exec_time=exec_time,
            max_ratio=None,
            max_pair=("", ""),
            mae=None,
        )
        return

def main_batch(
    input_dir: Path,
    output_dir: str,
    DRAW: str,
    show_plot: bool,
    mds_dim: int,
    time_limit_sec: float | None = None,
):
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"対象 CSV ファイル数: {len(csv_files)}")

    for csv_path in csv_files:
        print(f"\n=== {csv_path.name} を処理中 ===")
        run_for_one_csv(
            file_path=str(csv_path),
            output_dir=output_dir,
            show_plot=show_plot,
            DRAW=DRAW,
            mds_dim=mds_dim,
            time_limit_sec=time_limit_sec,
        )

    print("\nすべての CSV の処理が完了しました。")

if __name__ == "__main__":
    # 1つのファイルを実行する場合の場合の入力・出力フォルダ
    DEFAULT_CSV_PATH = '/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_data/3D（L=0~30）/n=50, k=5/manhattan_n50_k5_d2_add9.csv'
    OUTPUT_DIR_SINGLE = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm"

    # 複数のファイルを一括処理する場合の入力・出力フォルダ
    INPUT_DIR = Path("/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_dataのコピー")
    OUTPUT_DIR_BATCH = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/experiment/提案手法2/artificial_data/k, n=50"

    MODE = "single"  # "single" or "batch"
    DRAW = "normal"   # "normal" or "MDS"
    MDS_DIM = 2  # 2 or 3

    # 1ファイルあたりの時間制限（秒）。None にすると無制限
    TIME_LIMIT_SEC = 10 * 60   # 例: 3時間

    if MODE == "single":
        run_for_one_csv(
            file_path=DEFAULT_CSV_PATH,
            output_dir=OUTPUT_DIR_SINGLE,
            show_plot=False,
            DRAW=DRAW,
            mds_dim=MDS_DIM,
            time_limit_sec=TIME_LIMIT_SEC,
        )
    elif MODE == "batch":
        main_batch(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR_BATCH,
            DRAW=DRAW,
            show_plot=False,
            mds_dim=MDS_DIM,
            time_limit_sec=TIME_LIMIT_SEC,
        )