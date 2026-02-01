"""
マンハッタン距離空間の生成スクリプト

・n, k, d を入力し、[0, n-1]^d 上に k 個のランダムな点を配置
・L1（マンハッタン）距離行列を計算
・距離行列（CSV）と距離空間の図示（PDF）を出力
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Literal
import matplotlib
matplotlib.use("Agg")  # GUI環境がなくても動作するよう設定
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

plt.rcParams["axes.labelsize"] = 10   # 軸ラベルのサイズ
plt.rcParams["xtick.labelsize"] = 8   # 目盛り数字
plt.rcParams["ytick.labelsize"] = 8

# PDFにTrueTypeを埋め込む（文字化け/□回避）
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42

# 日本語を含む sans-serif を優先
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
    """
    与えられた点集合に対してマンハッタン距離行列を計算する関数

    Parameters
    ----------
    points : np.ndarray
        各行が1点を表す (k, d) 形状の配列

    Returns
    -------
    np.ndarray
        マンハッタン距離行列 (k, k)
    """
    diffs = points[:, None, :] - points[None, :, :]
    return np.abs(diffs).sum(axis=2)

def sample_points_on_grid(n: int, k: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    [0, n-1]^d の整数グリッド上から k 点を重複なしでランダム抽出する関数

    Parameters
    ----------
    n : int
        各座標方向の範囲（0〜n-1）
    k : int
        サンプル点数
    d : int
        次元数
    seed : int, optional
        乱数シード（再現性確保用）

    Returns
    -------
    np.ndarray
        抽出された k 点の座標（整数） shape=(k, d)
    """
    if n <= 0 or d <= 0 or k <= 0:
        raise ValueError("n, k, d はすべて正の整数である必要があります。")
    total = n ** d
    if k > total:
        raise ValueError(f"k={k} はグリッド総点数 n^d={total} を超えています。")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(total, size=k, replace=False)
    points = np.column_stack([(idxs // (n ** p)) % n for p in range(d)])
    return points.astype(int)

def lift_2d_points_to_3d(points_2d: np.ndarray, z_value: int = 0) -> np.ndarray:
    """2次元点群 (k,2) を 3次元 (k,3) に持ち上げる。z座標は一様に z_value にする。"""
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
    """
    [0,n-1]^d の整数グリッド上から点を1つサンプルする。

    - d=3 のとき、x,y は常に [0, n-1] の一様整数から取る。
    - z_range が指定された場合、z は原則として [-z_range, +z_range] に収める。
    - forbid_z が指定された場合、その集合に含まれる z を避ける（d=3 のときのみ）。
    - z_dist:
        * "uniform": z を一様整数（[-z_range,+z_range]）から取る（z_range 必須）
        * "normal" : z ~ N(z_mu, z_sigma^2) を丸めて整数化し、z_range があればクリップする
    """
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

            # --- z を生成 ---
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

            # 禁止値を回避
            if forbid_z is not None and z in forbid_z:
                continue

            p = (x, y, z)

        else:
            p = tuple(int(x) for x in rng.integers(0, n, size=d))

        if p not in used:
            used.add(p)
            return np.array(p, dtype=int)

def classical_mds_from_dist(D: np.ndarray, m: int = 2) -> np.ndarray:
    """
    古典的MDS（Torgerson法）により距離行列Dを2次元空間に埋め込む。

    ※L1距離に対しては厳密な等距離ではないが、可視化目的に用いる。

    Parameters
    ----------
    D : np.ndarray
        対称な距離行列
    m : int, default=2
        出力次元数

    Returns
    -------
    np.ndarray
        埋め込み後の座標 (k, m)
    """
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
    """指定されたファイルパスのディレクトリが存在しなければ作成する"""
    dname = os.path.dirname(path)
    if dname and not os.path.exists(dname):
        os.makedirs(dname, exist_ok=True)

from mpl_toolkits.mplot3d import Axes3D  # 3D用

def plot_points(points: np.ndarray, labels: List[str], ax: plt.Axes, k: int, n: int):
    """
    点群の配置を1次元・2次元・3次元でプロットする。
    d > 3 の場合は先頭3軸を使用。
    k と n を受け取りタイトルに反映する。
    """

    d = points.shape[1]

    # --- 1次元 ---
    if d == 1:
        xs = points[:, 0]
        ax.scatter(xs, [0]*len(xs), color="blue")

        # ここでラベルを表示
        for i, lbl in enumerate(labels):
            ax.text(xs[i], 0.05, lbl, ha='center')

        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_title(f"{k} random points in a {n}×{n} Grid")

        # グリッド線（薄いグレー）
        ax.grid(True, linestyle='-', color='lightgray')

    # --- 2次元 ---
    elif d == 2:
        xs, ys = points[:, 0], points[:, 1]
        ax.scatter(xs, ys, color="blue")

        # ここでラベルを表示
        for i, lbl in enumerate(labels):
            ax.text(xs[i], ys[i] + 0.05, lbl, ha='center')

        ax.set_aspect('equal')
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title(f"{k} random points in a {n}×{n} Grid")

        # グリッド線（薄いグレーの実線）
        ax.grid(True, linestyle='-', color='lightgray')

    # --- 3次元 ---
    else:
        # matplotlib の 3D Axes の場合：ax は Axes3D
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

        ax.scatter(xs, ys, zs, color="blue")

        # ここでラベルを表示
        for i, lbl in enumerate(labels):
            ax.text(xs[i], ys[i], zs[i], lbl)

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_title(f"{k} random points in a {n}×{n}×{n} Grid")

        # 3Dグリッドも薄いグレーの実線
        ax.grid(True, linestyle='-', color='lightgray')
        
        for label in ax.get_zticklabels():
            label.set_fontsize(8)

def plot_heatmap(D: np.ndarray, labels: List[str], ax: plt.Axes):
    """距離行列をヒートマップとして描画"""
    cax = ax.imshow(D, interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("マンハッタン距離行列")
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

def plot_mds(D: np.ndarray, labels: List[str], ax: plt.Axes):
    """距離行列から得られた2次元MDS埋め込みを描画"""
    emb = classical_mds_from_dist(D, m=2)
    ax.scatter(emb[:, 0], emb[:, 1])
    for i, lbl in enumerate(labels):
        ax.text(emb[i, 0], emb[i, 1] + 0.02, lbl, ha='center')
    ax.set_title("距離に基づく2次元埋め込み（MDS）")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

@dataclass
class ManhattanMetricResult:
    """結果をまとめるデータ構造"""
    points: np.ndarray
    labels: List[str]
    D: np.ndarray
    csv_path: str
    pdf_path: str

def generate_manhattan_metric_single(n: int, k: int, d: int, seed: Optional[int], sample_index: int, out_dir: str) -> ManhattanMetricResult:
    """
    指定パラメータに基づいてマンハッタン距離空間を1つ生成。
    sample_index は 1 始まりの連番（_s{index} をファイル名に付与）。
    距離行列（CSV）と図示（PDF）を出力する。
    """
    
    points = sample_points_on_grid(n, k, d, seed)
    labels = [f"v{i+1}" for i in range(k)]
    D = manhattan_distance_matrix(points)

    out_prefix = os.path.join(out_dir, f"manhattan_n{n}_k{k}_d{d}_s{sample_index}")
    csv_path = f"{out_prefix}.csv"
    pdf_path = f"{out_prefix}.pdf"
    ensure_dir(csv_path)

    # 距離行列をCSVに保存
    pd.DataFrame(D, index=labels, columns=labels).astype(int).to_csv(csv_path, encoding="utf-8")
    
    # --- PDF 出力 ---
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
    """
    同一 n,k,d で m 個のサンプルを連続生成。
    ・被りチェックは行わない（高速）
    ・シードを与えた場合は base_seed から連番で決定（再現性あり）
    """
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
    """
    2次元の点集合を1つ生成して距離行列を保存し、その後「3次元となる点」を1つずつ追加して
    距離行列を保存する操作を L 回繰り返す。

    - まず 2次元 (d=2) の k 点を生成（CSV/PDF保存）
    - 次に、その点群を z=z_base の 3次元点群に持ち上げ
    - そこへ 3次元点を1点追加 → (k+1) 点の 3D L1距離行列を保存
    - さらにもう1点追加 → (k+2) 点の 3D L1距離行列を保存
    - ... を L 回

    追加点の z は z_dist に従って生成されます（uniform / normal）。
    追加点が z=z_base の平面上に来ることは常に禁止します（z ≠ z_base）。

    seed を指定すると、同じデータ系列を再現できる。

    出力ファイル名は以下の形式:
      - base: .../manhattan_n{n}_k{k}_d2_base
      - step i: .../manhattan_n{n}_k{k}_d2_add{i}
    """
    if L < 0:
        raise ValueError("L は 0 以上の整数である必要があります。")

    # 追加点の z の範囲が未指定なら、自然なデフォルトとして n を使う
    if z_range is None:
        z_range = n

    rng = np.random.default_rng(seed)

    # --- まず 2D を生成して保存（rng による再現可能サンプリング） ---
    total = n ** 2
    idxs = rng.choice(total, size=k, replace=False)
    points2d = np.column_stack([(idxs // (n ** p)) % n for p in range(2)]).astype(int)

    labels2d = [f"v{i+1}" for i in range(k)]
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
        forbid = {int(z_base)}  # 追加点は常に z != z_base
        new_p = sample_unique_point_on_grid(
            n=n,
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
            plot_points(points3d, labels, ax1, k=k_now, n=n)
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
    out_dir = "/Users/yokoyamaedna0215/Library/CloudStorage/Box-Box/Personal/OTPM_algorithm/data/artificial_data/verify"

    mode = input(
        "モードを選択してください。\n"
        "  1: d=2 または d=3 のランダム点を m 個生成\n"
        "  2: 2Dを1つ生成→3D点を1つずつ追加してL回保存\n"
        "入力 (1/2): "
    ).strip() or "1"

    if mode == "1":
        n = int(input("グリッドの一辺の大きさ n を入力してください：").strip())
        k = int(input("サンプル点数 k を入力してください：").strip())
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
        n = int(input("グリッドの一辺の大きさ n を入力してください：").strip())
        k = int(input("2Dベースのサンプル点数 k を入力してください：").strip())
        L = int(input("追加する3D点の回数 L を入力してください：").strip())
        seed_input = input("乱数シード（任意・空欄で省略）: ").strip()
        seed = int(seed_input) if seed_input else None
        z_base = 0  # 2D点群は常に z=0 の平面に置く

        z_dist_input = input("追加点の z 分布（uniform / normal。未指定なら uniform）: ").strip().lower()
        z_dist = z_dist_input if z_dist_input in ("uniform", "normal") else "uniform"

        # --- 追加点 z のデフォルト設定 ---
        # 一般的な「2D平面からのずれ」を作るため、範囲は ±n にクリップ（uniform の範囲、normal のクリップ範囲）
        z_range = n

        # normal のときは 0 周りに集中しつつ、ほどよく散るようにスケールを n に合わせる
        z_mu = 0.0
        alpha = 0.2   # ← ここを 0.1, 0.2, 0.3, 0.5, 1.0 ... と変える
        z_sigma = max(1.0, alpha * n)

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
