
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# 1. CAU HINH CHUNG
# =====================================================================
ROOT = Path(__file__).resolve().parent          # duong dan tuong doi
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5],
                [-0.5, 2.0]])
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

N_TRAIN_POOL = 2000   # thi nghiem A lay n mau dau tien cua pool
N_VAL = 400
N_TEST = 100          # theo de bai
SEED_TRAIN, SEED_VAL, SEED_TEST = 42, 43, 44   # moi tap 1 seed rieng

SHOW_FIGURES = True   # True: hien cua so anh (van luu file); False: chi luu file

# Baseline de xuat cho cac thi nghiem A/B/C (TV2 co the chinh)
BASELINE = dict(seed=0, batch_size=32, epochs=300, optimizer="adam", lr=1e-2,
                activation="tanh", hidden_sizes=[16], patience=50,
                standardize_x=True)


# =====================================================================
# 2. HAM GAUSSIAN VA SINH DU LIEU
# =====================================================================
def gaussian_2d(X, mu=None, cov=None):
    """f(x) = 1/(2*pi*sqrt(det C)) * exp(-0.5 (x-mu)^T C^-1 (x-mu)).
    X: (N,2) hoac (2,). Tra ve (N,)."""
    mu = MU if mu is None else np.asarray(mu, dtype=float)
    cov = COV if cov is None else np.asarray(cov, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    diff = X - mu
    quad = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))


def make_split(n, seed):
    """Lay mau deu x1~U(-2,2), x2~U(-4,4); nhan y = f(x1,x2)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(*X1_RANGE, n), rng.uniform(*X2_RANGE, n)])
    return X, gaussian_2d(X)


def generate_all():
    """3 tap doc lap, moi tap 1 seed rieng."""
    return {"train_pool": make_split(N_TRAIN_POOL, SEED_TRAIN),
            "val": make_split(N_VAL, SEED_VAL),
            "test": make_split(N_TEST, SEED_TEST)}


def save_csv(out_dir=DATA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in generate_all().items():
        pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y}).to_csv(
            out_dir / f"{name}.csv", index=False)


def get_data(n_train=None, standardize=True, as_torch=False):
    """Nap du lieu co dinh. Train = n_train mau DAU TIEN cua pool (tap con
    long nhau). Chuan hoa x bang mean/std cua TRAIN (khong ro ri val/test).
    Tra ve dict: X_train, y_train, X_val, y_val, X_test, y_test, mean, std."""
    n_train = N_TRAIN_POOL if n_train is None else n_train
    if not 1 <= n_train <= N_TRAIN_POOL:
        raise ValueError(f"n_train phai trong [1, {N_TRAIN_POOL}]")
    sets = generate_all()
    X_tr, y_tr = sets["train_pool"][0][:n_train], sets["train_pool"][1][:n_train]
    X_va, y_va = sets["val"]
    X_te, y_te = sets["test"]
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    if standardize:
        X_tr, X_va, X_te = [(X - mean) / std for X in (X_tr, X_va, X_te)]
    out = dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
               X_test=X_te, y_test=y_te, mean=mean, std=std)
    if as_torch:
        import torch
        for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            t = torch.tensor(out[k], dtype=torch.float32)
            out[k] = t.unsqueeze(1) if k.startswith("y_") else t
    return out


# =====================================================================
# 3. KIEM TRA TINH DUNG DAN
# =====================================================================
def check_data():
    from scipy.stats import multivariate_normal
    print("=== KIEM TRA DU LIEU ===")
    print(f"det(C) = {np.linalg.det(COV):.4f}   (ky vong 0.35)")
    print("C^-1 =\n", np.linalg.inv(COV))
    print("C xac dinh duong:", bool(np.all(np.linalg.eigvalsh(COV) > 0)))
    peak = gaussian_2d([0, 0])[0]
    print(f"f(0,0) = {peak:.6f}   (ly thuyet {1/(2*np.pi*np.sqrt(0.35)):.6f})")

    sets = generate_all()
    X, y = sets["train_pool"]
    err = np.max(np.abs(y - multivariate_normal(MU, COV).pdf(X)))
    print(f"Sai lech toi da so voi scipy: {err:.2e}")
    assert err < 1e-12

    a, b = get_data(600), get_data(600)
    assert np.array_equal(a["X_train"], b["X_train"]), "khong tai lap"
    small = get_data(200, standardize=False)["X_train"]
    big = get_data(2000, standardize=False)["X_train"]
    assert np.array_equal(small, big[:200]), "tap con khong long nhau"
    for Xs, _ in sets.values():
        assert Xs[:, 0].min() >= -2 and Xs[:, 0].max() <= 2
        assert Xs[:, 1].min() >= -4 and Xs[:, 1].max() <= 4
    tr = {tuple(r) for r in sets["train_pool"][0]}
    assert not any(tuple(r) in tr for r in sets["test"][0]), "ro ri test"
    assert not any(tuple(r) in tr for r in sets["val"][0]), "ro ri val"
    print("Tat ca kiem tra DAT.\n")


# =====================================================================
# 4. TRUC QUAN HOA
# =====================================================================
def visualize_data():
    import matplotlib
    if not SHOW_FIGURES:
        matplotlib.use("Agg")   # chi luu file, khong mo cua so
    import matplotlib.pyplot as plt

    def _done(fig):
        if not SHOW_FIGURES:    # neu can hien anh thi giu hinh mo
            plt.close(fig)

    dpi = 200
    colors = {"train_pool": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    labels = {"train_pool": "Train (pool 2000)", "val": "Validation (400)",
              "test": "Test (100)"}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sets = generate_all()

    g1, g2 = np.linspace(*X1_RANGE, 100), np.linspace(*X2_RANGE, 100)
    G1, G2 = np.meshgrid(g1, g2)
    Z = gaussian_2d(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)

    # Hinh 1: scatter 3D
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    for k, (X, y) in sets.items():
        ax.scatter(X[:, 0], X[:, 1], y, s=6 if k == "train_pool" else 14,
                   c=colors[k], alpha=0.5 if k == "train_pool" else 0.9, label=labels[k])
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$y = f(x_1,x_2)$")
    ax.set_title("Hình 1. Scatter 3D của dữ liệu train/validation/test")
    ax.legend(loc="upper left"); ax.view_init(25, -60)
    fig.savefig(FIG_DIR / "fig1_scatter3d.png", dpi=dpi, bbox_inches="tight"); _done(fig)

    # Hinh 2: be mat ham that
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$f(x_1,x_2)$")
    ax.set_title("Hình 2. Bề mặt hàm Gaussian thật")
    fig.colorbar(surf, shrink=0.6, pad=0.1, label="f"); ax.view_init(28, -60)
    fig.savefig(FIG_DIR / "fig2_true_surface.png", dpi=dpi, bbox_inches="tight"); _done(fig)

    # Hinh 3: contour + mau train
    fig, ax = plt.subplots(figsize=(6, 7))
    cs = ax.contourf(G1, G2, Z, levels=15, cmap="viridis"); fig.colorbar(cs, label="f")
    Xt = sets["train_pool"][0]
    ax.scatter(Xt[:, 0], Xt[:, 1], s=2, c="white", alpha=0.35, label="Train samples")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Hình 3. Đường đồng mức của hàm thật và mẫu train")
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "fig3_contour.png", dpi=dpi, bbox_inches="tight"); _done(fig)

    # Hinh 4: histogram x1, x2, y
    Xa = np.vstack([s[0] for s in sets.values()])
    ya = np.concatenate([s[1] for s in sets.values()])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, data, name, ttl in zip(axs, (Xa[:, 0], Xa[:, 1], ya), ("$x_1$", "$x_2$", "$y$"),
                                   ("Phân bố $x_1$", "Phân bố $x_2$", "Phân bố $y$")):
        ax.hist(data, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel(name); ax.set_ylabel("Số mẫu"); ax.set_title(ttl)
    fig.suptitle("Hình 4. Histogram của $x_1$, $x_2$, $y$ (gộp train+val+test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_histograms.png", dpi=dpi, bbox_inches="tight"); _done(fig)

    # Bang thong ke
    rows = []
    for k, (X, y) in sets.items():
        for name, v in (("x1", X[:, 0]), ("x2", X[:, 1]), ("y", y)):
            rows.append(dict(set=k, variable=name, n=len(v), min=v.min(), max=v.max(),
                             mean=v.mean(), std=v.std(ddof=1), median=np.median(v)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "data_statistics.csv", index=False)
    print("=== THONG KE MO TA ===")
    print(df.round(4).to_string(index=False))
    y = sets["train_pool"][1]
    print(f"\nTi le mau train co y < 0.01: {(y < 0.01).mean():.1%}")
    print(f"Ti le mau train co y < 0.05: {(y < 0.05).mean():.1%}")
    print(f"Da luu hinh vao {FIG_DIR}")
    if SHOW_FIGURES:
        plt.show()   # hien ca 4 hinh; dong cua so de chuong trinh ket thuc


# =====================================================================
# 5. CHAY TOAN BO
# =====================================================================
def main():
    global SHOW_FIGURES
    if "--no-show" in sys.argv:
        SHOW_FIGURES = False
    save_csv()
    print(f"Da luu CSV vao {DATA_DIR}\n")
    check_data()
    visualize_data()
    print(f"Da luu thong ke vao {RESULTS_DIR}")


if __name__ == "__main__":
    main()
