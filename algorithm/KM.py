"""
karzanov_metric.py

Karzanov (1998) "Metrics with Finite Sets of Primitive Extensions" に出てくる
modular closure 構成と LG-graph 構成を、できるだけ忠実に実装したもの。

主な機能:
- KarzanovMetric: 有理距離 (Fraction) を扱う有限距離空間
- modular_closure(): (1.1), (1.2) による modular closure の構成
- lg_graph(): modular metric から LG-graph (least generating graph) を構成

頂点は整数 0,1,2,... をラベルとして用いる。
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Tuple, List, Optional, Iterable, Callable, Any

import itertools
import random
# (collections.Counterはverify系でのみ使用なので削除)
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import os
import json
import hashlib

from pathlib import Path

# ============================
# Metric isomorphism (isometry) checker (unlabeled)
# ============================

def _metric_signature_all_distances(m: "KarzanovMetric", v: Vertex) -> tuple[Fraction, ...]:
    """Vertex signature invariant under relabeling: sorted multiset of distances to all other vertices."""
    dists = [m.get(v, u) for u in m.V if u != v]
    return tuple(sorted(dists))

def is_isometric_unlabeled(
    m1: "KarzanovMetric",
    m2: "KarzanovMetric",
    tol: float = 0.0,
) -> bool:
    """Check whether two finite metrics are isometric up to relabeling of *all* vertices.

    - Exact mode (tol<=0): uses Fraction equality.
    - Tolerant mode (tol>0): compares by float within tol.

    This uses partitioning by vertex signatures (distance multisets) and DFS backtracking.
    It is intended for verify-mode / moderate |V|.
    """

    if len(m1.V) != len(m2.V):
        return False

    V1 = list(m1.V)
    V2 = list(m2.V)
    n = len(V1)

    if tol <= 0.0:
        sig1 = {v: _metric_signature_all_distances(m1, v) for v in V1}
        sig2 = {v: _metric_signature_all_distances(m2, v) for v in V2}
    else:
        # float signatures (rounded) for tolerant grouping
        def _sigf(m: "KarzanovMetric", v: Vertex) -> tuple[float, ...]:
            dists = [float(m.get(v, u)) for u in m.V if u != v]
            dists.sort()
            return tuple(dists)
        sig1 = {v: _sigf(m1, v) for v in V1}
        sig2 = {v: _sigf(m2, v) for v in V2}

    # Group by signature
    groups1: dict[Any, list[Vertex]] = {}
    groups2: dict[Any, list[Vertex]] = {}
    for v in V1:
        groups1.setdefault(sig1[v], []).append(v)
    for v in V2:
        groups2.setdefault(sig2[v], []).append(v)

    if set(groups1.keys()) != set(groups2.keys()):
        return False
    for k in groups1.keys():
        if len(groups1[k]) != len(groups2[k]):
            return False

    # Order groups small->large for faster DFS
    group_keys = sorted(groups1.keys(), key=lambda k: len(groups1[k]))

    mapping: dict[Vertex, Vertex] = {}
    used2: set[Vertex] = set()

    def _dist_ok(a1: Vertex, b1: Vertex, a2: Vertex, b2: Vertex) -> bool:
        x = m1.get(a1, b1)
        y = m2.get(a2, b2)
        if tol <= 0.0:
            return x == y
        return abs(float(x) - float(y)) <= tol

    def _consistent_new(v1: Vertex, v2: Vertex) -> bool:
        # check distances to already mapped vertices
        for u1, u2 in mapping.items():
            if not _dist_ok(v1, u1, v2, u2):
                return False
        return True

    def dfs(gidx: int) -> bool:
        if gidx == len(group_keys):
            return True

        key = group_keys[gidx]
        A = groups1[key]
        B = groups2[key]

        # local ordering by remaining candidates
        A_order = sorted(A, key=lambda v: 0)  # stable

        # precompute candidate lists
        cand: dict[Vertex, list[Vertex]] = {}
        for v1 in A_order:
            cand[v1] = [v2 for v2 in B if (v2 not in used2 and _consistent_new(v1, v2))]
            if not cand[v1]:
                return False

        A_order = sorted(A_order, key=lambda v: len(cand[v]))

        def dfs_group(i: int) -> bool:
            if i == len(A_order):
                return dfs(gidx + 1)
            v1 = A_order[i]
            for v2 in cand[v1]:
                if v2 in used2:
                    continue
                if not _consistent_new(v1, v2):
                    continue
                mapping[v1] = v2
                used2.add(v2)
                if dfs_group(i + 1):
                    return True
                used2.remove(v2)
                del mapping[v1]
            return False

        return dfs_group(0)

    return dfs(0)

# --- Helper functions for time limit ---
def _deadline_from_limit(start_time: float, time_limit_sec: float | None) -> float | None:
    return (start_time + time_limit_sec) if (time_limit_sec is not None) else None

def _check_timeout(deadline: float | None):
    if deadline is not None and time.time() > deadline:
        raise TimeoutError("Time limit exceeded for this task.")

import networkx as nx

# ==== Proposal-method-2-like evaluation helper functions ====

def _total_graph_length(G: nx.Graph) -> float:
    total = 0.0
    for _u, _v, data in G.edges(data=True):
        w = data.get("weight", None)
        if w is None:
            w = float(data.get("length", 0.0))
        total += float(w)
    return float(total)

def _compute_error_metrics_for_original_vertices(
    G: nx.Graph,
    mat: List[List[float]],
    n_original: int,
) -> tuple[float, tuple[int, int], float]:
    """元の頂点(0..n_original-1)間の最短路距離を入力 mat と比較し、
    (max_abs_rel_error, argmax_pair(i,j), mae) を返す。
    """
    dist_mat = nx.floyd_warshall_numpy(G, weight="weight")
    max_abs_rel = -1.0
    argmax = (0, 0)
    abs_sum = 0.0
    cnt = 0

    for i in range(n_original):
        for j in range(i + 1, n_original):
            d_real = float(dist_mat[i, j])
            d_orig = float(mat[i][j])
            abs_sum += abs(d_real - d_orig)
            cnt += 1
            if d_orig != 0.0:
                rel = (d_real - d_orig) / d_orig
                a = abs(rel)
                if a > max_abs_rel:
                    max_abs_rel = a
                    argmax = (i, j)

    mae = (abs_sum / cnt) if cnt > 0 else 0.0
    if max_abs_rel < 0.0:
        max_abs_rel = 0.0
        argmax = (0, 0)
    return max_abs_rel, argmax, mae

def save_evaluation_summary_row_karzanov(
    output_dir: str,
    input_file: str,
    status: str,
    original_n: int | None,
    final_n: int | None,
    final_m: int | None,
    added_vertices: int | None,
    total_length: float | None,
    exec_time_sec: float,
    max_rel_error_abs: float | None,
    max_pair: tuple[int, int] | None,
    mae: float | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "実行結果.csv")
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "input_file",
                "status",
                "original_n",
                "final_n",
                "final_m",
                "added_vertices",
                "total_length",
                "exec_time_sec",
                "max_rel_error_abs",
                "max_error_pair_i",
                "max_error_pair_j",
                "mae",
            ])

        def _fmt_opt(x):
            return "" if x is None else x

        mp0 = ""
        mp1 = ""
        if max_pair is not None:
            mp0, mp1 = max_pair

        w.writerow([
            os.path.basename(input_file),
            status,
            _fmt_opt(original_n),
            _fmt_opt(final_n),
            _fmt_opt(final_m),
            _fmt_opt(added_vertices),
            "" if total_length is None else float(total_length),
            f"{exec_time_sec:.6f}",
            "" if max_rel_error_abs is None else float(max_rel_error_abs),
            mp0,
            mp1,
            "" if mae is None else float(mae),
        ])

# ========== Step log/diff helpers ==========


def append_verify_trial_summary_csv(
    out_csv_path: str,
    *,
    trial: int,
    seed: int,
    isometric: bool,
    steps: int,
    final_n: int,
    sum_uv_changed: int,
    sum_uv_argmax_added: int,
    first_diverge_step: str,
    first_diverge_field: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    file_exists = os.path.exists(out_csv_path)
    with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "trial",
                "seed",
                "is_isometric",
                "steps",
                "final_n",
                "sum_uv_changed_by_added",
                "sum_uv_argmax_from_added",
                "first_diverge_step",
                "first_diverge_field",
            ])
        w.writerow([
            int(trial),
            int(seed),
            1 if isometric else 0,
            int(steps),
            int(final_n),
            int(sum_uv_changed),
            int(sum_uv_argmax_added),
            first_diverge_step,
            first_diverge_field,
        ])

def _sha1_of_int_list(xs: list[int]) -> str:
    h = hashlib.sha1()
    for x in xs:
        h.update(str(int(x)).encode("utf-8"))
        h.update(b",")
    return h.hexdigest()

def _sha1_of_fraction_list(xs: list[Fraction]) -> str:
    h = hashlib.sha1()
    for x in xs:
        # stable exact representation
        h.update(str(x.numerator).encode("utf-8"))
        h.update(b"/")
        h.update(str(x.denominator).encode("utf-8"))
        h.update(b",")
    return h.hexdigest()

def summarize_step_log_diff(log1: list[dict[str, Any]], log2: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a concise summary of the first diverging step between two closure construction logs."""
    n1 = len(log1)
    n2 = len(log2)
    m = min(n1, n2)
    first = None
    for i in range(m):
        a = log1[i]
        b = log2[i]
        keys = [
            "triple_canon",
            "dv_sha1",
            "remaining_sha1",
        ]
        for k in keys:
            if a.get(k) != b.get(k):
                first = (i, k, a.get(k), b.get(k))
                break
        if first is not None:
            break

    out: dict[str, Any] = {
        "len_log1": n1,
        "len_log2": n2,
        "first_diverge": None,
    }

    if first is None:
        if n1 != n2:
            out["first_diverge"] = {
                "step_index": m,
                "field": "length",
                "log1": n1,
                "log2": n2,
            }
        return out

    i, field, va, vb = first
    out["first_diverge"] = {
        "step_index": i + 1,  # 1-based for human
        "field": field,
        "log1": va,
        "log2": vb,
        "log1_step": log1[i],
        "log2_step": log2[i],
    }
    return out

Vertex = int
Edge = Tuple[Vertex, Vertex]

def _display_label(v: Vertex, n_original: int, labels_original: list[str] | None = None) -> str:
    """表示用ラベル。

    - 元頂点: CSV ラベルがあればそれを使う（labels_original[v]）
    - そうでなければ 0,1,2,...
    - modular closure で追加された頂点: m0,m1,m2,...

    内部の頂点ID(int)は変えず、表示・保存のラベルだけを変える。
    """
    if v < n_original:
        if labels_original is not None and 0 <= v < len(labels_original):
            return str(labels_original[v])
        return str(v)
    return f"m{v - n_original}"
def read_distance_matrix_and_labels_from_csv(path: str) -> tuple[list[list[float]], list[str]]:
    """距離行列とラベルを CSV から読み込む。

    - ヘッダーありの場合: 1行目の列ラベルと各行の行ラベルを読み、整合していればそれを採用。
    - ヘッダーなしの場合: ラベルは "0","1",... を自動生成。

    戻り値: (mat, labels)
    """
    rows: list[list[str]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r or all(cell.strip() == "" for cell in r):
                continue
            rows.append([cell.strip() for cell in r])

    if not rows:
        raise ValueError(f"CSV が空です: {path}")

    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    first = rows[0]
    has_header = not all(_is_float(x) for x in first)

    mat: list[list[float]] = []
    labels: list[str] = []

    if has_header:
        col_labels = [x for x in first[1:]]
        data_rows = rows[1:]
        row_labels = []
        for r in data_rows:
            if len(r) < 2:
                continue
            row_labels.append(r[0])
            nums = [float(x) for x in r[1:]]
            mat.append(nums)

        n = len(mat)
        if any(len(row) != n for row in mat):
            raise ValueError(f"正方行列ではありません: {path}, shape = {[len(r) for r in mat]}")

        # prefer row labels; if both present and consistent, keep them
        if col_labels and len(col_labels) == n and row_labels and len(row_labels) == n:
            if col_labels == row_labels:
                labels = row_labels
            else:
                # fallback to row labels (more reliable for display)
                labels = row_labels
        else:
            labels = [str(i) for i in range(n)]

    else:
        for r in rows:
            nums = [float(x) for x in r]
            mat.append(nums)

        n = len(mat)
        if any(len(row) != n for row in mat):
            raise ValueError(f"正方行列ではありません: {path}, shape = {[len(r) for r in mat]}")
        labels = [str(i) for i in range(n)]

    return mat, labels

@dataclass
class KarzanovMetric:
    """
    有理距離を保持する有限距離空間。

    - V: 頂点集合（整数ラベル）
    - d: (u,v) -> Fraction, 対称で対角 0 とする
    """
    V: List[Vertex]
    d: Dict[Tuple[Vertex, Vertex], Fraction]

    # ========= 基本操作 =========

    @classmethod
    def from_matrix(cls, mat: List[List[float | int]]) -> "KarzanovMetric":
        """
        NxN の距離行列から KarzanovMetric を作成。
        mat[i][i] は 0 を仮定。
        """
        n = len(mat)
        V = list(range(n))
        d: Dict[Tuple[Vertex, Vertex], Fraction] = {}
        for i in range(n):
            for j in range(i + 1, n):
                val = Fraction(mat[i][j]).limit_denominator()
                d[(i, j)] = val
        return cls(V=V, d=d)

    def copy(self) -> "KarzanovMetric":
        return KarzanovMetric(self.V.copy(), self.d.copy())

    def get(self, u: Vertex, v: Vertex) -> Fraction:
        """距離 m(u,v) を返す。対角は 0。"""
        if u == v:
            return Fraction(0)
        if (u, v) in self.d:
            return self.d[(u, v)]
        if (v, u) in self.d:
            return self.d[(v, u)]
        raise KeyError(f"distance not defined for pair ({u},{v})")

    def set(self, u: Vertex, v: Vertex, value: Fraction) -> None:
        """距離 m(u,v) を設定（対称）。"""
        if u == v:
            assert value == 0, "diagonal must be 0"
            return
        if value < 0:
            raise ValueError("metric distances must be nonnegative")
        a, b = (u, v) if u < v else (v, u)
        self.d[(a, b)] = Fraction(value)

    # ========= median・median-less triple =========

    def is_median(self, s0: Vertex, s1: Vertex, s2: Vertex, v: Vertex) -> bool:
        """
        v が {s0,s1,s2} の median かどうか判定。
        m(s_i, v) + m(v, s_j) = m(s_i, s_j) を全ての 0<=i<j<=2 で満たすか。
        """
        d = self.get
        triples = [(s0, s1), (s0, s2), (s1, s2)]
        for a, b in triples:
            if d(a, v) + d(v, b) != d(a, b):
                return False
        return True

    def has_median(self, s0: Vertex, s1: Vertex, s2: Vertex) -> bool:
        """{s0,s1,s2} に median が存在するか。"""
        for v in self.V:
            if self.is_median(s0, s1, s2, v):
                return True
        return False

    def find_medianless_triple(self, deadline: float | None = None, rng: Optional[random.Random] = None) -> Optional[Tuple[Vertex, Vertex, Vertex]]:
        """
        median を持たない triple {s0,s1,s2} を 1 つ探す。
        なければ None。
        V の順序は rng が与えられていればシャッフルされる。
        """
        V = list(self.V)
        if rng is not None:
            rng.shuffle(V)
        n = len(V)
        for i in range(n):
            _check_timeout(deadline)
            for j in range(i + 1, n):
                _check_timeout(deadline)
                for k in range(j + 1, n):
                    _check_timeout(deadline)
                    a, b, c = V[i], V[j], V[k]
                    if not self.has_median(a, b, c):
                        return (a, b, c)
        return None

    def find_medianless_triple_avoiding(
        self,
        forbidden: Optional[Tuple[Vertex, Vertex, Vertex]],
        deadline: float | None = None,
    ) -> Optional[Tuple[Vertex, Vertex, Vertex]]:
        """median を持たない triple を 1 つ探す。ただし forbidden と同一の triple は避ける。

        verify の高速 trial 用。全列挙(list_medianless_triples)は重いので、走査しながら最初に見つかったものを返す。
        forbidden が None のときは find_medianless_triple と同じ。
        """
        if forbidden is None:
            return self.find_medianless_triple(deadline=deadline)

        V = self.V
        n = len(V)
        fset = tuple(sorted(forbidden))
        for i in range(n):
            _check_timeout(deadline)
            for j in range(i + 1, n):
                _check_timeout(deadline)
                for k in range(j + 1, n):
                    _check_timeout(deadline)
                    a, b, c = V[i], V[j], V[k]
                    if tuple(sorted((a, b, c))) == fset:
                        continue
                    if not self.has_median(a, b, c):
                        return (a, b, c)
        return None

    def find_medianless_triple_avoiding_set(
        self,
        forbidden: set[Tuple[Vertex, Vertex, Vertex]],
        rng: Optional[random.Random] = None,
        random_tries: int = 200,
        deadline: float | None = None,
    ) -> Optional[Tuple[Vertex, Vertex, Vertex]]:
        """median を持たない triple を 1 つ探す（forbidden 集合を避ける）。

        verify の高速 trial 用（速度最優先）。
        - 乱択で random_tries 回だけ候補を拾って試す（trial 間の多様性も稼ぐ）
        - 見つからなければ None を返す（全走査フォールバックはしない）

        forbidden は (a,b,c) を昇順ソートした canonical 形で入れておく。
        戻り値も canonical (a<b<c) で返す。
        """
        V = self.V
        n = len(V)
        if n < 3:
            return None

        if rng is None or random_tries <= 0:
            return None

        for _ in range(random_tries):
            _check_timeout(deadline)
            a, b, c = rng.sample(V, 3)
            tri = tuple(sorted((a, b, c)))
            if tri in forbidden:
                continue
            if not self.has_median(tri[0], tri[1], tri[2]):
                return tri

        return None

    def list_medianless_triples(self, max_count: Optional[int] = None, deadline: float | None = None) -> List[Tuple[Vertex, Vertex, Vertex]]:
        """median を持たない triple を列挙（必要なら max_count で打ち切り）。"""
        V = self.V
        n = len(V)
        out: List[Tuple[Vertex, Vertex, Vertex]] = []
        for i in range(n):
            _check_timeout(deadline)
            for j in range(i + 1, n):
                _check_timeout(deadline)
                for k in range(j + 1, n):
                    _check_timeout(deadline)
                    a, b, c = V[i], V[j], V[k]
                    if not self.has_median(a, b, c):
                        out.append((a, b, c))
                        if max_count is not None and len(out) >= max_count:
                            return out
        return out

    # ========= primitive extension (1.1), (1.2) =========

    def _add_primitive_extension(
        self,
        triple: Tuple[Vertex, Vertex, Vertex],
        rng: Optional[random.Random] = None,
        randomize_fill: bool = False,
        log_u_pick: bool = False,
        step_log: Optional[list[dict[str, Any]]] = None,
        fill_strategy: Optional[Callable[[List[Vertex], "KarzanovMetric", Tuple[Vertex, Vertex, Vertex]], List[Vertex]]] = None,
    ) -> Vertex:
        """
        論文の (1.1), (1.2) に従って、median-less triple {s0,s1,s2} に対する
        primitive extension を 1 回行う。

        戻り値: 追加された新頂点 v のラベル。
        """
        s0, s1, s2 = triple
        base_vertices = list(self.V)  # v 追加前の頂点集合を保存
        d = self.get

        # 新しい頂点ラベル v を決める（最大ラベル+1）
        v = (max(base_vertices) + 1) if base_vertices else 0
        self.V.append(v)

        # (1.1) による m(v,s_i)
        s = [s0, s1, s2]
        for i in range(3):
            si = s[i]
            sj = s[(i + 1) % 3]
            sk = s[(i + 2) % 3]
            val = (d(si, sj) + d(si, sk) - d(sj, sk)) / 2
            if val < 0:
                raise ValueError("primitive extension produced negative length (check input metric)")
            self.set(v, si, val)

        # (1.2) による m(v,u) for u ∈ V\{triple}
        # W: 既に m(v,·) が定義されている点集合（v はまだ入れない）
        W = {s0, s1, s2}
        W0 = set(W)  # 初期の W（triple の3点）
        uv_changed_by_added = 0  # max が W0 だけの場合と異なる回数
        uv_argmax_from_added = 0  # argmax が W0 以外（後から入った点）になる回数
        # triple に含まれない既存頂点
        remaining = [u for u in base_vertices if u not in (s0, s1, s2)]

        # Decide remaining order
        if fill_strategy is not None:
            remaining = list(fill_strategy(remaining, self, (s0, s1, s2)))
        elif randomize_fill and rng is not None and len(remaining) >= 2:
            rng.shuffle(remaining)

        # Compact per-step info (no long logs)
        remaining_sha1 = _sha1_of_int_list([int(x) for x in remaining])

        for u in remaining:
            # (1.2) m(u,v) := max{ m(u,x) - m(x,v) : x ∈ W }  （論文通り：絶対値なし）
            # 解析用：初期W0(=triple) だけの max と、現在の W の max を比較する。

            # max over W0
            max_w0: Fraction | None = None
            for x in W0:
                val = d(u, x) - self.get(v, x)
                if max_w0 is None or val > max_w0:
                    max_w0 = val
            if max_w0 is None:
                max_w0 = Fraction(0)

            # max over current W (and remember argmax)
            max_val: Fraction | None = None
            argmax_x: Vertex | None = None
            for x in W:
                val = d(u, x) - self.get(v, x)
                if max_val is None or val > max_val:
                    max_val = val
                    argmax_x = x

            if max_val is None:
                # W should never be empty (it starts with the triple), but keep safe.
                max_val = Fraction(0)

            # Counters: did added points in W affect the result?
            if max_val != max_w0:
                uv_changed_by_added += 1
            if argmax_x is not None and argmax_x not in W0:
                uv_argmax_from_added += 1

            if max_val < 0:
                # In the theory this should not happen for a valid metric extension.
                raise ValueError(
                    f"primitive extension produced negative m(u,v) in (1.2): u={u}, v={v}, max_val={max_val}"
                )

            self.set(u, v, max_val)
            # この時点で m(u,v) が定義されたので、次の u' のためには u も W に加えてよい
            W.add(u)

        if step_log is not None:
            triple_canon = tuple(sorted((int(s0), int(s1), int(s2))))
            # distances from new vertex v to all base vertices in increasing id order
            base_sorted = sorted(int(x) for x in base_vertices)
            dv_list = [self.get(v, int(u)) for u in base_sorted]
            step_log.append({
                "step": len(step_log) + 1,
                "new_vertex": int(v),
                "triple": (int(s0), int(s1), int(s2)),
                "triple_canon": triple_canon,
                "randomize_fill": bool(randomize_fill),
                "remaining_len": int(len(remaining)),
                "uv_changed_by_added": int(uv_changed_by_added),
                "uv_argmax_from_added": int(uv_argmax_from_added),
                "remaining_sha1": remaining_sha1,
                "dv_sha1": _sha1_of_fraction_list(dv_list),
                # keep small heads for human inspection
                "remaining_head": [int(x) for x in remaining[:12]],
                "dv_head": [str(x) for x in dv_list[:12]],
            })

        return v

    # ========= modular closure =========

    def modular_closure(
        self,
        verbose: bool = False,
        chooser: Optional[Callable[[List[Tuple[Vertex, Vertex, Vertex]], "KarzanovMetric"], Tuple[Vertex, Vertex, Vertex]]] = None,
        trace: Optional[List[Tuple[Vertex, Vertex, Vertex]]] = None,
        max_candidates: Optional[int] = None,
        deadline: float | None = None,
        rng: Optional[random.Random] = None,
        randomize_fill: bool = False,
        log_u_pick: bool = False,
        step_log: Optional[list[dict[str, Any]]] = None,
        fill_strategy: Optional[Callable[[List[Vertex], "KarzanovMetric", Tuple[Vertex, Vertex, Vertex]], List[Vertex]]] = None,
    ) -> "KarzanovMetric":
        """
        Karzanov の手続きに従って modular closure を構成する。
        median-less triple が存在しなくなるまで primitive extension を繰り返す。

        - chooser が None のときは従来通り「見つかった最初の triple」を使う。
        - chooser が与えられているときは、候補 triple のリストから chooser が選んだものを使う。
        - trace が与えられているときは、各 step で選ばれた triple を trace に append する。
        - max_candidates は候補列挙の打ち切り（高速化用）。None のときは全列挙。
        """
        step = 0
        while True:
            _check_timeout(deadline)
            if chooser is None:
                triple = self.find_medianless_triple(deadline=deadline, rng=rng)
            else:
                triples = self.list_medianless_triples(max_count=max_candidates, deadline=deadline)
                if not triples:
                    triple = None
                else:
                    # chooser は rng を closure 側で束縛して渡す
                    triple = chooser(triples, self)

            if triple is None:
                if verbose:
                    print(f"[modular_closure] finished after {step} extensions, |V|={len(self.V)}")
                break

            step += 1
            if trace is not None:
                trace.append(triple)
            if verbose:
                print(f"[modular_closure] step {step}: extend on triple {triple}")
            self._add_primitive_extension(
                triple,
                rng=rng,
                randomize_fill=randomize_fill,
                log_u_pick=log_u_pick,
                step_log=step_log,
                fill_strategy=fill_strategy,
            )

        return self

# ============================
# LG-graph (least generating graph)
# ============================

    def lg_graph(self, deadline: float | None = None):
        """
        modular metric m から LG-graph G=(V,E) を構成する。

        定義: 完全グラフ K_V から、
        「ある z∈V−{x,y} に対して m(x,z)+m(z,y)=m(x,y) を満たすような辺 xy」
        をすべて削除したものが LG-graph。
        """

        G = nx.Graph()
        G.add_nodes_from(self.V)
        V = self.V
        d = self.get

        for i in range(len(V)):
            _check_timeout(deadline)
            for j in range(i + 1, len(V)):
                _check_timeout(deadline)
                u, v = V[i], V[j]
                uv = d(u, v)
                redundant = False
                for z in V:
                    _check_timeout(deadline)
                    if z == u or z == v:
                        continue
                    if d(u, z) + d(z, v) == uv:
                        redundant = True
                        break
                if not redundant:
                    # weight は float と Fraction の両方を持たせておく
                    G.add_edge(u, v, weight=float(uv), length=uv)

        return G

# remaining order strategies for (1.2)

def fill_identity(remaining: List[Vertex], m: "KarzanovMetric", triple: Tuple[Vertex, Vertex, Vertex]) -> List[Vertex]:
    return list(remaining)

def read_distance_matrix_from_csv(path: str) -> List[List[float]]:
    """
    距離行列を CSV から読み込む簡易関数。

    想定フォーマット:
    - N x N の正方行列
    - ヘッダーなし:
        1行目から N行目までが距離行列
    - ヘッダーあり:
        1行目: 空白 or ラベル, 2列目以降が列ラベル
        2行目以降: 1列目が行ラベル, 2列目以降が距離

    ラベルは無視し、数値部分だけを返す。
    """
    rows: List[List[str]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            # 空行はスキップ
            if not r or all(cell.strip() == "" for cell in r):
                continue
            rows.append([cell.strip() for cell in r])

    if not rows:
        raise ValueError(f"CSV が空です: {path}")

    # ヘッダーかどうかを判定
    first = rows[0]
    # 1行目の 2 列目以降が全部数値なら「ヘッダーなし」とみなす
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not all(_is_float(x) for x in first)

    mat: List[List[float]] = []
    if has_header:
        # 1行目は列ラベルなので飛ばす
        data_rows = rows[1:]
        for r in data_rows:
            # 1列目が行ラベルなので飛ばす
            nums = [float(x) for x in r[1:]]
            mat.append(nums)
    else:
        # 全行をそのまま数値行として読む
        for r in rows:
            nums = [float(x) for x in r]
            mat.append(nums)

    # 正方行列チェックは軽くしておく
    n = len(mat)
    if any(len(row) != n for row in mat):
        raise ValueError(f"正方行列ではありません: {path}, shape = {[len(r) for r in mat]}")

    return mat

def save_metric_as_csv(
    m: KarzanovMetric,
    input_file: str,
    output_dir: str,
    suffix: str = "",
    as_float: bool = False,
    with_header: bool = True,
    n_original: int | None = None,
    labels_original: list[str] | None = None,
) -> str:
    """modular closure 後の距離行列（metric）を CSV で保存する。"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, base_name + suffix + "_MC.csv")

    V = list(m.V)
    label = (lambda x: _display_label(x, n_original, labels_original)) if (n_original is not None) else (lambda x: str(x))

    def fmt(x: Fraction) -> str:
        if as_float:
            return str(float(x))
        return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if with_header:
            w.writerow([""] + [label(x) for x in V])
        for u in V:
            row = [label(u)] if with_header else []
            for v in V:
                row.append(fmt(m.get(u, v)))
            w.writerow(row)

    return output_file

def save_network_as_nexus(
    G,
    input_file: str,
    output_dir: str,
    suffix: str = "",
    n_original: int | None = None,
    labels_original: list[str] | None = None,
) -> str:
    """G を NEXUS (SplitsTree/ネットワーク系で扱える簡易フォーマット) で保存してパスを返す。"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, base_name + suffix + ".nex")

    nodes = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes, start=1)}
    label = (lambda x: _display_label(x, n_original, labels_original)) if (n_original is not None) else (lambda x: str(x))

    nexus_str = "#NEXUS\n\nBEGIN NETWORK;\n"
    nexus_str += f"DIMENSIONS nVertices={len(nodes)} nEdges={G.number_of_edges()};\n"

    nexus_str += "VERTICES"
    for node in nodes:
        nid = node_to_id[node]
        nexus_str += f"\n    id={nid} label='{label(node)}',"
    nexus_str = nexus_str.rstrip(",")
    nexus_str += "\n;\nEDGES"

    for i, (u, v, data) in enumerate(G.edges(data=True), start=1):
        # karzanov.py では weight(float) と length(Fraction) の両方を持たせている
        w = data.get("length", data.get("weight", 1.0))
        u_id = node_to_id[u]
        v_id = node_to_id[v]
        nexus_str += f"\n    id={i} sid={u_id} tid={v_id} weight={float(w)},"
    nexus_str = nexus_str.rstrip(",")
    nexus_str += "\n;\nEND; [NETWORK]\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(nexus_str)

    return output_file

def save_graph_as_pdf(
    G,
    input_file: str,
    output_dir: str,
    suffix: str = "",
    n_original: int | None = None,
    labels_original: list[str] | None = None,
) -> str:
    """LG-graph を PDF で保存してパスを返す（提案手法1/2と同じ見た目に寄せる）。"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, base_name + suffix + "_LG.pdf")

    # 提案手法1/2と同じく Kamada-Kawai に固定
    pos = nx.kamada_kawai_layout(G)

    label = (lambda x: _display_label(x, n_original, labels_original)) if (n_original is not None) else (lambda x: str(x))
    labels = {node: label(node) for node in G.nodes()}

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=30,
        font_size=5,
        width=0.5,
    )

    # 辺ラベル（length があればそれ、なければ weight）
    edge_lengths = nx.get_edge_attributes(G, "length")
    if edge_lengths:
        edge_labels = {e: float(w) for e, w in edge_lengths.items()}
    else:
        edge_w = nx.get_edge_attributes(G, "weight")
        edge_labels = {e: float(w) for e, w in edge_w.items()}

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=5,
        )

    plt.title("LG-graph with Weighted Edges")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file

def run_once(
    mat: List[List[float]],
    verbose: bool = True,
    draw: bool = True,
    draw_mode: str = "normal",
    mds_dim: int = 2,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_pdf: bool = True,
    save_nexus: bool = True,
    save_mc_csv: bool = True,
    mc_csv_as_float: bool = False,
    save_suffix: str = "",
    time_limit_sec: float | None = None,
    labels_original: list[str] | None = None,
) -> None:
    """1つの距離行列について modular_closure と LG-graph を作り、必要なら保存も行う。"""
    start_time = time.time()
    deadline = _deadline_from_limit(start_time, time_limit_sec)
    try:
        m = KarzanovMetric.from_matrix(mat)
        n_original = len(m.V)
        labels_original = labels_original if labels_original is not None else [str(i) for i in range(n_original)]
        if verbose:
            print("original |V|:", len(m.V))

        m.modular_closure(verbose=verbose, chooser=None, trace=None, max_candidates=None, deadline=deadline, rng=None, randomize_fill=False, log_u_pick=False)
        if verbose:
            print("after modular_closure |V|:", len(m.V))

        _check_timeout(deadline)
        if input_file is not None and output_dir is not None and save_mc_csv:
            mc_path = save_metric_as_csv(
                m,
                input_file=input_file,
                output_dir=output_dir,
                suffix=save_suffix,
                as_float=mc_csv_as_float,
                with_header=True,
                n_original=n_original,
                labels_original=labels_original,
            )
            if verbose:
                print("MC-CSV で保存:", mc_path)

        G = m.lg_graph(deadline=deadline)
        exec_time = time.time() - start_time
        if verbose:
            print("LG-graph: |V|, |E| =", G.number_of_nodes(), G.number_of_edges())
            print(f"処理時間: {exec_time:.2f} 秒")

        # 指標（元頂点間の距離誤差、MAE、全長）
        max_abs_rel_error, max_pair, mae = _compute_error_metrics_for_original_vertices(G, mat, n_original)
        total_length = _total_graph_length(G)
        if verbose:
            print(f"グラフ全長: {total_length}")
            print(f"最大相対誤差（絶対値） |Δd|/d_orig = {max_abs_rel_error:.6g} at [{max_pair[0]}][{max_pair[1]}]")
            print(f"平均絶対誤差 (MAE): {mae}")

        added_vertices = int(G.number_of_nodes() - n_original)

        # 保存（input_file と output_dir が与えられているときのみ）
        _check_timeout(deadline)
        if input_file is not None and output_dir is not None:
            dm_save = (draw_mode or "normal").lower().strip()

            # In MDS mode, we do not save the normal LG layout PDF (it clutters outputs).
            # The MDS PDF will be produced in the MDS drawing block (even when draw=False).
            if save_pdf and dm_save not in {"mds", "mds2", "mds3"}:
                pdf_path = save_graph_as_pdf(
                    G,
                    input_file=input_file,
                    output_dir=output_dir,
                    suffix=save_suffix,
                    n_original=n_original,
                    labels_original=labels_original,
                )
                if verbose:
                    print("PDF で保存:", pdf_path)

            if save_nexus:
                nex_path = save_network_as_nexus(
                    G,
                    input_file=input_file,
                    output_dir=output_dir,
                    suffix=save_suffix,
                    n_original=n_original,
                    labels_original=labels_original,
                )
                if verbose:
                    print("NEXUS で保存:", nex_path)

        # 画面描画（draw_mode に応じて 2D/3D/MDS 切替）
        _check_timeout(deadline)
        if draw or (input_file is not None and output_dir is not None and save_pdf and (draw_mode or "").lower().strip() in {"mds", "mds2", "mds3"}):
            dm = (draw_mode or "normal").lower().strip()

            if dm in {"mds", "mds3", "mds2"}:
                from sklearn.manifold import MDS

                if mds_dim not in (2, 3):
                    raise ValueError(f"mds_dim must be 2 or 3, got {mds_dim}")

                dist_mat = nx.floyd_warshall_numpy(G, weight="weight")
                mds = MDS(
                    n_components=mds_dim,
                    dissimilarity="precomputed",
                    random_state=0,
                )
                coords = mds.fit_transform(dist_mat)
                nodes = list(G.nodes())
                idx_of = {node: idx for idx, node in enumerate(nodes)}
                # --- Axis alignment (PCA) ---
                # MDS coordinates are only determined up to rigid transforms.
                # Align principal axes to (x,y[,z]) so the plot doesn't look arbitrarily rotated.
                coords = coords - coords.mean(axis=0, keepdims=True)
                # SVD-based PCA rotation
                _u, _s, _vt = np.linalg.svd(coords, full_matrices=False)
                coords = coords @ _vt.T

                # Optional: fix sign to make orientation deterministic
                for d in range(coords.shape[1]):
                    if coords[:, d].sum() < 0:
                        coords[:, d] *= -1

                # --- Extra: rotate so dominant edge direction becomes axis-aligned (2D only) ---
                # PCA can still leave a Manhattan-like grid tilted (often ~45°).
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

                label = lambda x: _display_label(x, n_original, labels_original)

                if mds_dim == 2:
                    plt.figure(figsize=(8, 6))
                    for idx, node in enumerate(nodes):
                        plt.scatter(coords[idx, 0], coords[idx, 1], s=10, color='skyblue')
                        plt.text(coords[idx, 0], coords[idx, 1], label(node), fontsize=6)

                    for (u, v, _data) in G.edges(data=True):
                        i = idx_of[u]
                        j = idx_of[v]
                        plt.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            linewidth=0.5,
                            color='gray',
                        )

                    plt.title("LG-graph with Weighted Edges (MDS)")
                    plt.tight_layout()
                    if draw:
                        plt.show()
                    else:
                        if input_file is not None and output_dir is not None and save_pdf:
                            base_name = os.path.splitext(os.path.basename(input_file))[0]
                            output_file = os.path.join(output_dir, base_name + save_suffix + "_MDS.pdf")
                            plt.savefig(output_file)
                            plt.close()

                else:  # mds_dim == 3
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection="3d")

                    for idx, node in enumerate(nodes):
                        ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=10, color='skyblue')
                        ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2], label(node), fontsize=6)

                    for (u, v, _data) in G.edges(data=True):
                        i = idx_of[u]
                        j = idx_of[v]
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            [coords[i, 2], coords[j, 2]],
                            linewidth=0.5,
                            color='gray',
                        )

                    ax.set_title("LG-graph with Weighted Edges (MDS)")
                    plt.tight_layout()
                    if draw:
                        plt.show()
                    else:
                        if input_file is not None and output_dir is not None and save_pdf:
                            base_name = os.path.splitext(os.path.basename(input_file))[0]
                            output_file = os.path.join(output_dir, base_name + save_suffix + "_MDS3.pdf")
                            plt.savefig(output_file)
                            plt.close()

            # normal は A-4 の置換済みブロックが使われる
            if dm == "normal":
                plt.figure(figsize=(8, 6))

                # 提案手法1/2と同じく Kamada-Kawai に固定
                pos = nx.kamada_kawai_layout(G)

                label = lambda x: _display_label(x, n_original, labels_original)
                labels = {node: label(node) for node in G.nodes()}

                nx.draw(
                    G,
                    pos,
                    with_labels=True,
                    labels=labels,
                    node_size=30,
                    font_size=5,
                    width=0.5,
                )

                edge_lengths = nx.get_edge_attributes(G, "length")
                if edge_lengths:
                    edge_labels = {e: float(w) for e, w in edge_lengths.items()}
                else:
                    edge_w = nx.get_edge_attributes(G, "weight")
                    edge_labels = {e: float(w) for e, w in edge_w.items()}

                if edge_labels:
                    nx.draw_networkx_edge_labels(
                        G,
                        pos,
                        edge_labels=edge_labels,
                        font_size=5,
                    )

                plt.title("LG-graph with Weighted Edges")
                plt.axis("equal")
                plt.tight_layout()
                plt.show()

        # 実行結果.csv（提案手法2形式に寄せる）
        if input_file is not None and output_dir is not None:
            save_evaluation_summary_row_karzanov(
                output_dir=output_dir,
                input_file=input_file,
                status="OK",
                original_n=n_original,
                final_n=int(G.number_of_nodes()),
                final_m=int(G.number_of_edges()),
                added_vertices=added_vertices,
                total_length=total_length,
                exec_time_sec=exec_time,
                max_rel_error_abs=max_abs_rel_error,
                max_pair=max_pair,
                mae=mae,
            )

        return

    except TimeoutError:
        exec_time = time.time() - start_time
        if verbose:
            print(f"[TIMEOUT] {os.path.basename(input_file) if input_file else ''} は制限時間超過のため中断しました。")

        if input_file is not None and output_dir is not None:
            save_evaluation_summary_row_karzanov(
                output_dir=output_dir,
                input_file=input_file,
                status="TIMEOUT",
                original_n=None,
                final_n=None,
                final_m=None,
                added_vertices=None,
                total_length=None,
                exec_time_sec=exec_time,
                max_rel_error_abs=None,
                max_pair=None,
                mae=None,
            )
        return

# ============================
# verify-mode runner
# ============================

def run_verify_mode(
    mat: List[List[float]],
    input_file: str,
    output_dir: str,
    max_trials: int = 200,
    seed0: int = 0,
    verbose: bool = True,
    draw: bool = False,
    draw_mode: str = "normal",
    mds_dim: int = 2,
    save_pdf: bool = True,
    save_nexus: bool = True,
    save_mc_csv: bool = True,
    mc_csv_as_float: bool = False,
    iso_tol: float = 0.0,
    time_limit_sec: float | None = None,
    randomize_triple: bool = True,
) -> None:
    """同じ入力から modular closure を複数試し、異なる closure を見つけたら保存して終了する。

    - mc1（基準）は medianless triple の探索順・選択ともに決定的（rng=None）。
    - trial 側（mc2）は randomize_triple の値で triple 探索順を切り替え可能。
      ON のとき triple 探索順を乱択、OFF のとき決定的（rng=None）。
    - (1.2) の埋め方（remaining の順序）は fill_strategy（shuffle）で変える。
    """

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    deadline = _deadline_from_limit(start_time, time_limit_sec)

    if verbose:
        print("[karzanov] RUN_MODE=verify")
        print(f"  trials    : {max_trials}")
        print("  difference-source: ONLY randomized fill order (u order) in (1.2)")
        print("  fill_mode : random-shuffle")
        print(f"  seed0     : {seed0}")
        print(f"  iso_tol   : {iso_tol}")
        print(f"  randomize_triple(trials): {randomize_triple}")

    # Reference closure (deterministic fill: always "top" order)
    m1 = KarzanovMetric.from_matrix(mat)
    n_original = len(m1.V)
    trace1: List[Tuple[Vertex, Vertex, Vertex]] = []
    step_log1: list[dict[str, Any]] = []
    m1.modular_closure(
        verbose=verbose,
        chooser=None,
        trace=trace1,
        max_candidates=None,
        deadline=deadline,
        rng=None,
        randomize_fill=False,
        log_u_pick=False,
        step_log=step_log1,
        fill_strategy=fill_identity,
    )

    # Save reference artifacts
    if input_file is not None and output_dir is not None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if save_mc_csv:
            save_metric_as_csv(
                m1,
                input_file=input_file,
                output_dir=output_dir,
                suffix="_mc1",
                as_float=mc_csv_as_float,
                with_header=True,
                n_original=n_original,
                labels_original=None,
            )
        G1 = m1.lg_graph(deadline=deadline)
        if save_pdf:
            save_graph_as_pdf(
                G1,
                input_file=input_file,
                output_dir=output_dir,
                suffix="_mc1",
                n_original=n_original,
                labels_original=None,
            )
        if save_nexus:
            save_network_as_nexus(G1, input_file=input_file, output_dir=output_dir, suffix="_mc1", n_original=n_original, labels_original=None)

        # Trial summary CSV (appended each trial)
        trial_summary_csv = os.path.join(output_dir, base_name + "_verify_trials.csv")

    # Trials: only randomized fill order is varied (shuffle remaining in (1.2))
    for t in range(1, max_trials + 1):
        _check_timeout(deadline)
        rng = random.Random(seed0 + t)

        if verbose:
            print(f"[verify] trial {t}/{max_trials}: seed={seed0 + t}")

        m2 = KarzanovMetric.from_matrix(mat)
        trace2: List[Tuple[Vertex, Vertex, Vertex]] = []
        step_log2: list[dict[str, Any]] = []

        # Randomize ONLY the remaining order in (1.2) via fill_strategy.
        def fill_shuffle(
            remaining: List[Vertex],
            _m: "KarzanovMetric",
            _triple: Tuple[Vertex, Vertex, Vertex],
            _rng: random.Random = rng,
        ) -> List[Vertex]:
            xs = list(remaining)
            if len(xs) >= 2:
                _rng.shuffle(xs)
            return xs

        fill_strategy = fill_shuffle

        # triple selection (trials): optionally randomized by rng (shuffle V order inside find_medianless_triple)
        rng_for_triple = rng if randomize_triple else None
        m2.modular_closure(
            verbose=False,
            chooser=None,
            trace=trace2,
            max_candidates=None,
            deadline=deadline,
            rng=rng_for_triple,
            randomize_fill=False,
            log_u_pick=False,
            step_log=step_log2,
            fill_strategy=fill_strategy,
        )

        # Trial-level small diagnostics (aggregate)
        sum_uv_changed = 0
        sum_uv_argmax_added = 0
        for r in step_log2:
            sum_uv_changed += int(r.get("uv_changed_by_added", 0) or 0)
            sum_uv_argmax_added += int(r.get("uv_argmax_from_added", 0) or 0)

        # first diverge info against mc1 (for human)
        diff_summary_preview = summarize_step_log_diff(step_log1, step_log2)
        fd = diff_summary_preview.get("first_diverge")
        if fd is None:
            fd_step = ""
            fd_field = ""
        else:
            fd_step = str(fd.get("step_index", ""))
            fd_field = str(fd.get("field", ""))

        # Check isometry (unlabeled)
        same = is_isometric_unlabeled(m1, m2, tol=iso_tol)
        try:
            # Save trial summary CSV every trial
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            trial_summary_csv = os.path.join(output_dir, base_name + "_verify_trials.csv")
            append_verify_trial_summary_csv(
                trial_summary_csv,
                trial=t,
                seed=seed0 + t,
                isometric=bool(same),
                steps=len(step_log2),
                final_n=len(m2.V),
                sum_uv_changed=sum_uv_changed,
                sum_uv_argmax_added=sum_uv_argmax_added,
                first_diverge_step=fd_step,
                first_diverge_field=fd_field,
            )
        except Exception as e:
            if verbose:
                print("[verify-log] failed to append trial summary:", e)

        if same:
            if verbose:
                print("[verify] same (isometric); continue")
            continue

        if verbose:
            print("[verify] FOUND different (non-isometric) modular closure")

        suffix = f"_mc2_trial{t:04d}"

        # Diff summary of construction dynamics
        diff_summary = summarize_step_log_diff(step_log1, step_log2)
        if verbose:
            fd = diff_summary.get("first_diverge")
            if fd is None:
                print("[verify-diff] step logs look identical (but closures non-isometric).")
            else:
                print("[verify-diff] first diverging step:")
                print(f"  step={fd.get('step_index')} field={fd.get('field')}")
                a = fd.get("log1_step", {})
                b = fd.get("log2_step", {})
                print(f"  mc1 triple_canon={a.get('triple_canon')} remaining_sha1={str(a.get('remaining_sha1',''))[:8]} dv_sha1={str(a.get('dv_sha1',''))[:8]}")
                print(f"  mc2 triple_canon={b.get('triple_canon')} remaining_sha1={str(b.get('remaining_sha1',''))[:8]} dv_sha1={str(b.get('dv_sha1',''))[:8]}")
                print(f"  mc1 remaining_head={a.get('remaining_head')}")
                print(f"  mc2 remaining_head={b.get('remaining_head')}")

        # Save logs and diff summary (JSON)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        log_path1 = os.path.join(output_dir, base_name + "_mc1_buildlog.json")
        log_path2 = os.path.join(output_dir, base_name + suffix + "_buildlog.json")
        diff_path = os.path.join(output_dir, base_name + suffix + "_diff.json")
        try:
            with open(log_path1, "w", encoding="utf-8") as f:
                json.dump(step_log1, f, ensure_ascii=False, indent=2)
            with open(log_path2, "w", encoding="utf-8") as f:
                json.dump(step_log2, f, ensure_ascii=False, indent=2)
            with open(diff_path, "w", encoding="utf-8") as f:
                json.dump(diff_summary, f, ensure_ascii=False, indent=2)
            if verbose:
                print("[verify-diff] saved:", os.path.basename(log_path1), os.path.basename(log_path2), os.path.basename(diff_path))
        except Exception as e:
            if verbose:
                print("[verify-diff] failed to save diff logs:", e)


        G2 = m2.lg_graph(deadline=deadline)

        if save_mc_csv:
            save_metric_as_csv(
                m2,
                input_file=input_file,
                output_dir=output_dir,
                suffix=suffix,
                as_float=mc_csv_as_float,
                with_header=True,
                n_original=n_original,
                labels_original=None,
            )
        if save_pdf:
            save_graph_as_pdf(
                G2,
                input_file=input_file,
                output_dir=output_dir,
                suffix=suffix,
                n_original=n_original,
                labels_original=None,
            )
        if save_nexus:
            save_network_as_nexus(G2, input_file=input_file, output_dir=output_dir, suffix=suffix, n_original=n_original, labels_original=None)

        if draw:
            plt.figure(figsize=(8, 6))
            pos = nx.kamada_kawai_layout(G2)
            nx.draw(G2, pos, with_labels=True, node_size=30, font_size=5, width=0.5)
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        return

    print(f"[verify] different (non-isometric) modular closure was NOT found within max_trials={max_trials}.")

def main_batch(
    input_dir: Path,
    output_dir: str,
    verbose: bool = True,
    draw: bool = False,
    draw_mode: str = "normal",
    mds_dim: int = 2,
    save_pdf: bool = True,
    save_nexus: bool = True,
    save_mc_csv: bool = True,
    mc_csv_as_float: bool = False,
    time_limit_sec: float | None = None,
) -> None:
    in_path = input_dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_path.glob("*.csv"))
    if not csv_files:
        print(f"[main_batch] CSV が見つかりません: {in_path}")
        return

    for csv_path in csv_files:
        print(f"\n=== {csv_path.name} を処理中 ===")
        try:
            mat, labels = read_distance_matrix_and_labels_from_csv(str(csv_path))
        except Exception as e:
            print("CSV 読み込みでエラーが発生しました：", e)
            continue
        try:
            run_once(
                mat,
                verbose=verbose,
                draw=draw,
                draw_mode=draw_mode,
                mds_dim=mds_dim,
                input_file=str(csv_path),
                output_dir=str(out_path),
                save_pdf=save_pdf,
                save_nexus=save_nexus,
                time_limit_sec=time_limit_sec,
                save_mc_csv=save_mc_csv,
                mc_csv_as_float=mc_csv_as_float,
                labels_original=labels,
            )
        except TimeoutError:
            print(f"[TIMEOUT] {csv_path.name} は制限時間超過のためスキップして次へ進みます。")
            continue

def main():
    # ============================
    # ユーザー設定（ここだけ編集）
    # ============================

    # 実験モード: "batch" / "single" / "verify"（(1.2) の u 順序のみを変えて探索）
    RUN_MODE = "verify"

    # 入力（single のときに使用）
    INPUT_CSV_PATH = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_data/verify/chatgpt2.csv"

    # 入力（batch のときに使用）
    INPUT_DIR = Path("/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_dataのコピー2/3D（L=0~30）")

    # 出力先（PDF/NEXUS を保存）
    OUTPUT_DIR = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/experiment/karzanov/verify"

    VERBOSE = True # ログを出力するかどうか
    DRAW = False
    DRAW_MODE = "MDS"  # "normal" or "MDS"
    MDS_DIM = 3           # 2 or 3 (when DRAW_MODE=="MDS")
    SAVE_PDF = True
    SAVE_NEXUS = True
    SAVE_MC_CSV = True # Modular closure 後の csv 保存
    MC_CSV_AS_FLOAT = True # True：floatで保存、false：分数で保存

    # 1ファイルあたりの時間制限（秒）。None にすると無制限
    TIME_LIMIT_SEC = None#10 * 60

    # verify モード用（RUN_MODE=="verify" のときのみ使用）
    VERIFY_MAX_TRIALS = 5000
    VERIFY_SEED0 = 0
    VERIFY_ISO_TOL = 0.0
    VERIFY_RANDOMIZE_TRIPLE = True  # trial 側（mc2）の medianless triple 探索順を乱択する

    # ============================
    # 実行
    # ============================

    if RUN_MODE == "batch":
        print("[karzanov] RUN_MODE=batch")
        print("  input_dir :", INPUT_DIR)
        print("  output_dir:", OUTPUT_DIR)
        main_batch(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            verbose=VERBOSE,
            draw=DRAW,
            draw_mode=DRAW_MODE,
            mds_dim=MDS_DIM,
            save_pdf=SAVE_PDF,
            save_nexus=SAVE_NEXUS,
            time_limit_sec=TIME_LIMIT_SEC,
            save_mc_csv=SAVE_MC_CSV,
            mc_csv_as_float=MC_CSV_AS_FLOAT,
        )
        return

    if RUN_MODE == "verify":
        print("[karzanov] RUN_MODE=verify")
        print("  input_csv :", INPUT_CSV_PATH)
        print("  output_dir:", OUTPUT_DIR)
        mat, labels = read_distance_matrix_and_labels_from_csv(INPUT_CSV_PATH)
        run_verify_mode(
            mat,
            input_file=INPUT_CSV_PATH,
            output_dir=OUTPUT_DIR,
            max_trials=VERIFY_MAX_TRIALS,
            seed0=VERIFY_SEED0,
            verbose=VERBOSE,
            draw=DRAW,
            draw_mode=DRAW_MODE,
            mds_dim=MDS_DIM,
            save_pdf=SAVE_PDF,
            save_nexus=SAVE_NEXUS,
            save_mc_csv=SAVE_MC_CSV,
            mc_csv_as_float=MC_CSV_AS_FLOAT,
            iso_tol=VERIFY_ISO_TOL,
            time_limit_sec=TIME_LIMIT_SEC,
            randomize_triple=VERIFY_RANDOMIZE_TRIPLE,
        )
        return

    if RUN_MODE == "single":
        print("[karzanov] RUN_MODE=single")
        print("  input_csv :", INPUT_CSV_PATH)
        print("  output_dir:", OUTPUT_DIR)
        mat, labels = read_distance_matrix_and_labels_from_csv(INPUT_CSV_PATH)
        run_once(
            mat,
            verbose=VERBOSE,
            draw=DRAW,
            draw_mode=DRAW_MODE,
            mds_dim=MDS_DIM,
            input_file=INPUT_CSV_PATH,
            output_dir=OUTPUT_DIR,
            save_pdf=SAVE_PDF,
            save_nexus=SAVE_NEXUS,
            time_limit_sec=TIME_LIMIT_SEC,
            save_mc_csv=SAVE_MC_CSV,
            mc_csv_as_float=MC_CSV_AS_FLOAT,
            labels_original=labels,
        )
        return

    raise ValueError(f"RUN_MODE must be 'batch' or 'single' or 'verify' but got: {RUN_MODE}")

if __name__ == "__main__":
    main()