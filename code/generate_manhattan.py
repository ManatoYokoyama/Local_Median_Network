"""
generate_manhattan.py

マンハッタン距離（L1）に基づく人工距離行列データを生成するスクリプトです。

出力
- 距離行列: CSV
- 図: PDF（点配置 / 距離行列ヒートマップ / MDS）

使い方
- `python generate_manhattan.py` を実行し、対話入力でモードを選択します。
- `out_dir` はご自身の環境に合わせてパスを設定してください（下の main() 内）。

注意
- 出力形式や計算内容は変えず、コードを読みやすくするためにコメント/説明を整理しています。
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Literal
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Hiragino Sans",            # macOS 標準（推奨）
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",                # Windows
    "Meiryo",                   # Windows
    "Noto Sans CJK JP",         # Google
    "IPAexGothic",              # IPAex
    "Source Han Sans JP",       # 角ゴシック
    "DejaVu Sans"               # 最後のフォールバック
]

def manhattan_distance_matrix(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    return np.abs(diffs).sum(axis=2)

def sample_points_on_grid(n: int, k: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    if n <= 0 or k <= 0 or d <= 0:
        raise ValueError("n, k, d はすべて正の整数である必要があります。")
    total = k ** d
    if n > total:
        raise ValueError(f"n={n} はグリッド総点数 k^d={total} を超えています。")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(total, size=n, replace=False)
    points = np.column_stack([(idxs // (k ** p)) % k for p in range(d)])
    return points.astype(int)

def lift_2d_points_to_3d(points_2d: np.ndarray, z_value: int = 0) -> np.ndarray:
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d は shape=(k,2) の配列である必要があります。")
    z = np.full((points_2d.shape[0], 1), int(z_value), dtype=int)
    return np.concatenate([points_2d.astype(int), z], axis=1)

def sample_unique_point_on_grid(
    n: int,
    d: int,
    used: set[tuple[int, ...]],
    rng: np.random.Generator,
    z_range: Optional[int] = None,
    forbid_z: Optional[set[int]] = None,
    z_dist: Literal["uniform", "normal"] = "uniform",
    z_mu: float = 0.0,
    z_sigma: float = 1.0,
    max_tries: int = 100000,
) -> np.ndarray:
    if n <= 0 or d <= 0:
        raise ValueError("n, d は正の整数である必要があります。")

    tries = 0
    while True:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("サンプリングが収束しませんでした。条件（z_range/forbid_z/重複）を緩めてください。")

        if d == 3:
            x = int(rng.integers(0, n))
            y = int(rng.integers(0, n))

            if z_dist == "uniform":
                if z_range is None:
                    raise ValueError("z_dist='uniform' の場合は z_range を指定してください。")
                if z_range == 0:
                    raise ValueError("z_range=0 の場合、z の候補が 0 のみになるためサンプリングできません。")
                z = int(rng.integers(-z_range, z_range + 1))

            elif z_dist == "normal":
                if z_sigma <= 0:
                    raise ValueError("z_dist='normal' の場合、z_sigma は正である必要があります。")
                z = int(np.rint(rng.normal(loc=z_mu, scale=z_sigma)))
                if z_range is not None:
                    z = int(np.clip(z, -z_range, z_range))

            else:
                raise ValueError(f"未知の z_dist: {z_dist}")

            if forbid_z is not None and z in forbid_z:
                continue

            p = (x, y, z)

        else:
            p = tuple(int(x) for x in rng.integers(0, n, size=d))

        if p not in used:
            used.add(p)
            return np.array(p, dtype=int)

def classical_mds_from_dist(D: np.ndarray, m: int = 2) -> np.ndarray:
    k = D.shape[0]
    J = np.eye(k) - np.ones((k, k)) / k
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_clipped = np.clip(vals[:m], 0, None)
    return vecs[:, :m] @ np.diag(np.sqrt(vals_clipped))

def ensure_dir(path: str):
    dname = os.path.dirname(path)
    if dname and not os.path.exists(dname):
        os.makedirs(dname, exist_ok=True)

from mpl_toolkits.mplot3d import Axes3D  # 3D用

def plot_points(points: np.ndarray, labels: List[str], ax: plt.Axes, k: int, n: int):
    d = points.shape[1]

    # --- 1次元 ---
    if d == 1:
        xs = points[:, 0]
        ax.scatter(xs, [0]*len(xs), color="blue")

        for i, lbl in enumerate(labels):
            ax.text(xs[i], 0.05, lbl, ha='center')

        ax.set_yticks([])
        ax.set_xlabel("x座標")
        ax.set_title(f"{n}点を {k}×{k} グリッドからサンプル")

        ax.grid(True, linestyle='-', color='lightgray')

    # --- 2次元 ---
    elif d == 2:
        xs, ys = points[:, 0], points[:, 1]
        ax.scatter(xs, ys, color="blue")

        for i, lbl in enumerate(labels):
            ax.text(xs[i], ys[i] + 0.05, lbl, ha='center')

        ax.set_aspect('equal')
        ax.set_xlabel("x座標")
        ax.set_ylabel("y座標")
        ax.set_title(f"{n}点を {k}×{k} グリッドからサンプル")

        ax.grid(True, linestyle='-', color='lightgray')

    # --- 3次元 ---
    else:
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

        ax.scatter(xs, ys, zs, color="blue")

        for i, lbl in enumerate(labels):
            ax.text(xs[i], ys[i], zs[i], lbl)

        ax.set_xlabel("x座標")
        ax.set_ylabel("y座標")
        ax.set_zlabel("z座標")
        ax.set_title(f"{n}点を {k}×{k}×{k} グリッドからサンプル")

        ax.grid(True, linestyle='-', color='lightgray')
        
        for label in ax.get_zticklabels():
            label.set_fontsize(8)

def plot_heatmap(D: np.ndarray, labels: List[str], ax: plt.Axes):
    cax = ax.imshow(D, interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("マンハッタン距離行列")
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

def plot_mds(D: np.ndarray, labels: List[str], ax: plt.Axes):
    emb = classical_mds_from_dist(D, m=2)
    ax.scatter(emb[:, 0], emb[:, 1])
    for i, lbl in enumerate(labels):
        ax.text(emb[i, 0], emb[i, 1] + 0.02, lbl, ha='center')
    ax.set_title("距離に基づく2次元埋め込み（MDS）")
    ax.set_xlabel("第1軸")
    ax.set_ylabel("第2軸")

@dataclass
class ManhattanMetricResult:
    points: np.ndarray
    labels: List[str]
    D: np.ndarray
    csv_path: str
    pdf_path: str

def generate_manhattan_metric_single(n: int, k: int, d: int, seed: Optional[int], sample_index: int, out_dir: str) -> ManhattanMetricResult:
    points = sample_points_on_grid(n, k, d, seed)
    labels = [f"v{i+1}" for i in range(n)]
    D = manhattan_distance_matrix(points)

    out_prefix = os.path.join(out_dir, f"manhattan_n{n}_k{k}_d{d}_s{sample_index}")
    csv_path = f"{out_prefix}.csv"
    pdf_path = f"{out_prefix}.pdf"
    ensure_dir(csv_path)

    pd.DataFrame(D, index=labels, columns=labels).astype(int).to_csv(csv_path, encoding="utf-8")
    
    with PdfPages(pdf_path) as pdf:
        # 1ページ目：点配置図
        if d <= 2:
            fig2, ax2 = plt.subplots()
            plot_points(points, labels, ax2, k=k, n=n)
        else:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            plot_points(points, labels, ax2, k=k, n=n)

        pdf.savefig(fig2, bbox_inches=None)
        plt.close(fig2)

        # 2ページ目：距離行列ヒートマップ
        fig3, ax3 = plt.subplots()
        plot_heatmap(D, labels, ax3)
        pdf.savefig(fig3, bbox_inches=None)
        plt.close(fig3)

        # 3ページ目：MDS埋め込み
        fig4, ax4 = plt.subplots()
        plot_mds(D, labels, ax4)
        pdf.savefig(fig4, bbox_inches=None)
        plt.close(fig4)

    return ManhattanMetricResult(points, labels, D, csv_path, pdf_path)

def generate_manhattan_metric_multiple(n: int, k: int, d: int, m: int, out_dir: str, base_seed: Optional[int] = None) -> list[ManhattanMetricResult]:
    results = []
    for i in range(m):
        seed_i = None if base_seed is None else int(base_seed) + i
        res = generate_manhattan_metric_single(n, k, d, seed=seed_i, sample_index=i + 1, out_dir=out_dir)
        results.append(res)
    return results

def generate_manhattan_metric_2d_then_add_3d(
    n: int,
    k: int,
    L: int,
    out_dir: str,
    seed: Optional[int] = None,
    z_base: int = 0,
    z_range: Optional[int] = None,
    z_dist: Literal["uniform", "normal"] = "uniform",
    z_mu: float = 0.0,
    z_sigma: float = 1.0,
) -> list[ManhattanMetricResult]:
    if L < 0:
        raise ValueError("L は 0 以上の整数である必要があります。")

    if z_range is None:
        z_range = k

    rng = np.random.default_rng(seed)

    total = k ** 2
    idxs = rng.choice(total, size=n, replace=False)
    points2d = np.column_stack([(idxs // (k ** p)) % k for p in range(2)]).astype(int)

    labels2d = [f"v{i+1}" for i in range(n)]
    D2d = manhattan_distance_matrix(points2d)

    out_prefix_base = os.path.join(
        out_dir,
        f"manhattan_n{n}_k{k}_d2_base",
    )
    csv_path_base = f"{out_prefix_base}.csv"
    pdf_path_base = f"{out_prefix_base}.pdf"
    ensure_dir(csv_path_base)

    pd.DataFrame(D2d, index=labels2d, columns=labels2d).astype(int).to_csv(csv_path_base, encoding="utf-8")

    with PdfPages(pdf_path_base) as pdf:
        # 1ページ目：2D点配置
        fig1, ax1 = plt.subplots()
        plot_points(points2d, labels2d, ax1, k=k, n=n)
        pdf.savefig(fig1, bbox_inches=None)
        plt.close(fig1)

        # 2ページ目：ヒートマップ
        fig2, ax2 = plt.subplots()
        plot_heatmap(D2d, labels2d, ax2)
        pdf.savefig(fig2, bbox_inches=None)
        plt.close(fig2)

        # 3ページ目：MDS
        fig3, ax3 = plt.subplots()
        plot_mds(D2d, labels2d, ax3)
        pdf.savefig(fig3, bbox_inches=None)
        plt.close(fig3)

    results: list[ManhattanMetricResult] = []
    results.append(ManhattanMetricResult(points2d, labels2d, D2d, csv_path_base, pdf_path_base))

    # --- 3Dに持ち上げ + 追加点をL回 ---
    points3d = lift_2d_points_to_3d(points2d, z_value=z_base)

    used3d: set[tuple[int, ...]] = set(tuple(map(int, row)) for row in points3d)

    for i in range(1, L + 1):
        forbid = {int(z_base)}
        new_p = sample_unique_point_on_grid(
            n=k,
            d=3,
            used=used3d,
            rng=rng,
            z_range=z_range,
            forbid_z=forbid,
            z_dist=z_dist,
            z_mu=z_mu,
            z_sigma=z_sigma,
        )
        points3d = np.vstack([points3d, new_p[None, :]])

        k_now = points3d.shape[0]
        labels = [f"v{j+1}" for j in range(k_now)]
        D = manhattan_distance_matrix(points3d)

        out_prefix = os.path.join(
            out_dir,
            f"manhattan_n{n}_k{k}_d2_add{i}",
        )
        csv_path = f"{out_prefix}.csv"
        pdf_path = f"{out_prefix}.pdf"
        ensure_dir(csv_path)

        pd.DataFrame(D, index=labels, columns=labels).astype(int).to_csv(csv_path, encoding="utf-8")

        with PdfPages(pdf_path) as pdf:
            # 1ページ目：3D点配置（2D点は z=z_base 面上）
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            plot_points(points3d, labels, ax1, k=k, n=k_now)
            pdf.savefig(fig1, bbox_inches=None)
            plt.close(fig1)

            # 2ページ目：ヒートマップ
            fig2, ax2 = plt.subplots()
            plot_heatmap(D, labels, ax2)
            pdf.savefig(fig2, bbox_inches=None)
            plt.close(fig2)

            # 3ページ目：MDS
            fig3, ax3 = plt.subplots()
            plot_mds(D, labels, ax3)
            pdf.savefig(fig3, bbox_inches=None)
            plt.close(fig3)

        results.append(ManhattanMetricResult(points3d.copy(), labels, D, csv_path, pdf_path))

    return results

def main():
    out_dir = "./out"
    ensure_dir(os.path.join(out_dir, "_dummy.txt"))

    mode = input(
        "モードを選択してください。\n"
        "  1: d=2 または d=3 のランダム点を m 個生成\n"
        "  2: 2Dを1つ生成→3D点を1つずつ追加してL回保存\n"
        "入力 (1/2): "
    ).strip() or "1"

    if mode == "1":
        n = int(input("点数 n を入力してください：").strip())
        k = int(input("グリッドの一辺の大きさ k を入力してください：").strip())
        d = int(input("次元数 d（2 または 3）を入力してください：").strip())
        if d not in (2, 3):
            raise ValueError("このモードでは d は 2 または 3 を指定してください。")
        m = int(input("生成するサンプル個数 m を入力してください：").strip())
        seed_input = input("ベース乱数シード（任意・空欄で省略。指定時は各サンプルに +i で割当）: ").strip()
        base_seed = int(seed_input) if seed_input else None

        results = generate_manhattan_metric_multiple(n, k, d, m, out_dir=out_dir, base_seed=base_seed)

        for r in results:
            print("CSV:", r.csv_path, " | PDF:", r.pdf_path)

    elif mode == "2":
        n = int(input("点数 n を入力してください：").strip())
        k = int(input("2Dベースのグリッド一辺の大きさ k を入力してください：").strip())
        L = int(input("追加する3D点の回数 L を入力してください：").strip())
        seed_input = input("乱数シード（任意・空欄で省略）: ").strip()
        seed = int(seed_input) if seed_input else None
        z_base = 0

        z_dist_input = input("追加点の z 分布（uniform / normal。未指定なら uniform）: ").strip().lower()
        z_dist = z_dist_input if z_dist_input in ("uniform", "normal") else "uniform"

        z_range = k

        z_mu = 0.0
        alpha = 0.2
        z_sigma = max(1.0, alpha * k)

        results = generate_manhattan_metric_2d_then_add_3d(
            n=n,
            k=k,
            L=L,
            out_dir=out_dir,
            seed=seed,
            z_base=z_base,
            z_range=z_range,
            z_dist=z_dist,
            z_mu=z_mu,
            z_sigma=z_sigma,
        )

        for r in results:
            print("CSV:", r.csv_path, " | PDF:", r.pdf_path)

    else:
        raise ValueError("モードは 1 または 2 を指定してください。")


if __name__ == "__main__":
    main()
