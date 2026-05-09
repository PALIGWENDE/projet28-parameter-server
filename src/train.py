"""
Projet 28 — Entrainement distribue + Visualisations
=====================================================
Lance l'entrainement asynchrone avec N workers et genere
les graphiques de convergence et de resultats.

Auteurs : Anas ELAYATTI & Steven KABRE
EMSI Marrakech — IIR S7 — 2024/2025
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parameter_server import ParameterServer, Worker

np.random.seed(42)

# ══════════════════════════════════════════════════════════
#  GENERATION DES DONNEES
# ══════════════════════════════════════════════════════════

def generate_data(n_samples: int = 1500, n_features: int = 10):
    """
    Genere un dataset de classification binaire synthetique.

    Les features representent des caracteristiques normalisees
    (ex : taille oreilles, longueur poils, taille corps...).
    Les vraies ponderation W_true sont celles que le modele doit retrouver.

    Args:
        n_samples  (int) : nombre d'exemples (defaut 1500)
        n_features (int) : dimension des features (defaut 10)

    Returns:
        X      (np.ndarray) : features normalisees  (n x d)
        y      (np.ndarray) : labels binaires        (n,)
        W_true (np.ndarray) : vrais poids            (d,)
    """
    W_true = np.array([0.8, -0.5, 0.3, -0.7, 0.6,
                       -0.2, 0.4, -0.3, 0.5, -0.6])
    X = np.random.randn(n_samples, n_features)
    z = X @ W_true + np.random.randn(n_samples) * 0.3
    y = (z > 0).astype(float)
    # Normalisation
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y, W_true


def split_for_workers(X: np.ndarray, y: np.ndarray, n_workers: int):
    """
    Divise le dataset en N parts egales, une par worker.

    Args:
        X        : features completes
        y        : labels complets
        n_workers: nombre de workers

    Returns:
        List de tuples (X_i, y_i) pour chaque worker
    """
    n     = len(X)
    chunk = n // n_workers
    splits = []
    for i in range(n_workers):
        start = i * chunk
        end   = start + chunk if i < n_workers - 1 else n
        splits.append((X[start:end], y[start:end]))
    return splits


# ══════════════════════════════════════════════════════════
#  ENTRAINEMENT
# ══════════════════════════════════════════════════════════

def train(n_workers: int = 3, n_epochs: int = 15,
          batch_size: int = 32, lr: float = 0.05,
          verbose: bool = True):
    """
    Lance l'entrainement distribue asynchrone.

    Args:
        n_workers  : nombre de workers paralleles
        n_epochs   : nombre d'epoques par worker
        batch_size : taille du mini-batch
        lr         : taux d'apprentissage
        verbose    : afficher le recap initial

    Returns:
        server  : ParameterServer avec les poids finaux
        workers : liste des Workers termines
        W_final : poids finaux appris
        W_true  : vrais poids (reference)
        X, y    : dataset complet
    """
    if verbose:
        print("\n" + "="*60)
        print("  PROJET 28 — PARAMETER SERVER")
        print("="*60)
        print(f"  Workers    : {n_workers}")
        print(f"  Epoques    : {n_epochs}")
        print(f"  Batch size : {batch_size}")
        print(f"  Lr (alpha) : {lr}")
        print("="*60 + "\n")

    # Donnees
    X, y, W_true = generate_data()
    if verbose:
        print(f"Dataset : {len(X)} exemples, {X.shape[1]} features")
        print(f"Classes : {int(y.sum())} positifs, {int((1-y).sum())} negatifs\n")

    splits = split_for_workers(X, y, n_workers)
    if verbose:
        for i, (Xi, _) in enumerate(splits):
            print(f"  Worker {i+1} : {len(Xi)} exemples")
        print()

    # Serveur
    server = ParameterServer(n_params=X.shape[1], lr=lr)

    # Workers
    workers = []
    for i, (Xi, yi) in enumerate(splits):
        w = Worker(
            worker_id=i + 1,
            server=server,
            X=Xi, y=yi,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        workers.append(w)

    # Lancement asynchrone
    print("Lancement de l'entrainement asynchrone...\n")
    t0 = time.time()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    elapsed = time.time() - t0

    # Evaluation finale
    W_final, _ = server.pull_weights()
    z           = X @ W_final
    pred        = 1 / (1 + np.exp(-z))
    accuracy    = np.mean((pred > 0.5) == y)

    print(f"\nEntrainement termine en {elapsed:.2f}s")
    print(f"Total gradients recus : {server.total_gradients}")
    print(f"Version finale        : v{server.version}")
    print(f"Precision finale      : {accuracy*100:.1f}%\n")

    return server, workers, W_final, W_true, X, y


# ══════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════

COLORS = ["#185FA5", "#BA7517", "#3B6D11", "#A32D2D", "#534AB7"]

def plot_all(server, workers, W_final, W_true, out_dir="resultats"):
    """
    Genere les 4 graphiques de resultats.

    Args:
        server   : ParameterServer post-entrainement
        workers  : liste des Workers
        W_final  : poids appris finaux
        W_true   : vrais poids de reference
        out_dir  : dossier de sortie

    Fichiers generes :
        01_convergence.png
        02_poids_appris.png
        03_architecture.png
        04_gradients_par_worker.png
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # ── Graphique 1 : Convergence ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Projet 28 — Convergence du Parameter Server",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for i, w in enumerate(workers):
        ax.plot(range(1, len(w.losses) + 1), w.losses,
                color=COLORS[i % len(COLORS)], linewidth=2,
                label=f"Worker {w.id}", marker="o", markersize=3)
    ax.set_xlabel("Epoque")
    ax.set_ylabel("Perte (Binary Cross-Entropy)")
    ax.set_title("Perte par worker")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    losses = server.loss_history
    if losses:
        win      = max(1, len(losses) // 20)
        smoothed = np.convolve(losses, np.ones(win) / win, mode="valid")
        ax.plot(smoothed, color="#185FA5", linewidth=2, label="Perte globale lissee")
        ax.fill_between(range(len(smoothed)), smoothed, alpha=0.15, color="#185FA5")
    ax.set_xlabel("Mise a jour serveur (#)")
    ax.set_ylabel("Perte")
    ax.set_title("Evolution de la perte globale")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "01_convergence.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    paths.append(p); print(f"  Sauvegarde : {p}")

    # ── Graphique 2 : Poids appris vs vrais ───────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(W_true); x = np.arange(n); bw = 0.35
    ax.bar(x - bw/2, W_true,  bw, label="Vrais poids W*",     color="#185FA5", alpha=0.85)
    ax.bar(x + bw/2, W_final, bw, label="Poids appris W_final", color="#BA7517", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i+1}" for i in range(n)])
    ax.set_ylabel("Valeur du poids")
    ax.set_title("Poids appris vs poids reels", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    for i in range(n):
        err = abs(W_final[i] - W_true[i])
        ax.annotate(f"{err:.2f}",
                    xy=(x[i], max(W_true[i], W_final[i]) + 0.03),
                    ha="center", fontsize=8, color="#555")
    plt.tight_layout()
    p = os.path.join(out_dir, "02_poids_appris.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    paths.append(p); print(f"  Sauvegarde : {p}")

    # ── Graphique 3 : Architecture ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Architecture du Parameter Server (Projet 28)",
                 fontweight="bold", fontsize=13, pad=15)

    srv = mpatches.FancyBboxPatch((3.5, 5.5), 3, 1.6,
                                   boxstyle="round,pad=0.15",
                                   facecolor="#E6F1FB", edgecolor="#185FA5", linewidth=2)
    ax.add_patch(srv)
    ax.text(5, 6.55, "Serveur de Parametres",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0C447C")
    ax.text(5, 6.1, f"W = {len(W_final)} poids  |  v{server.version}  |  lr={server.lr}",
            ha="center", va="center", fontsize=9, color="#185FA5")

    wpos  = [(1.0, 2.5), (3.8, 2.5), (6.6, 2.5)]
    wbg   = ["#FAEEDA", "#EAF3DE", "#FCEBEB"]
    wec   = ["#BA7517", "#3B6D11", "#A32D2D"]

    for i, w in enumerate(workers[:3]):
        cx, cy = wpos[i]
        box = mpatches.FancyBboxPatch((cx, cy), 2.2, 1.5,
                                       boxstyle="round,pad=0.12",
                                       facecolor=wbg[i], edgecolor=wec[i], linewidth=1.5)
        ax.add_patch(box)
        ax.text(cx+1.1, cy+1.05, f"Worker {w.id}",
                ha="center", va="center", fontsize=10, fontweight="bold", color=wec[i])
        ax.text(cx+1.1, cy+0.6, f"{len(w.X)} exemples",
                ha="center", va="center", fontsize=8.5, color="#555")
        ax.text(cx+1.1, cy+0.25, f"{w.gradients_sent} gradients",
                ha="center", va="center", fontsize=8, color="#888")
        ax.annotate("", xy=(5, 5.5), xytext=(cx+1.1, cy+1.5),
                    arrowprops=dict(arrowstyle="->", color="#185FA5", lw=1.5,
                                   connectionstyle="arc3,rad=0.1"))
        ax.annotate("", xy=(cx+1.1, cy+1.5), xytext=(5, 5.5),
                    arrowprops=dict(arrowstyle="->", color="#3B6D11", lw=1.2,
                                   connectionstyle="arc3,rad=-0.15"))

    ax.text(5, 1.5,
            f"Asynchrone  |  Total gradients : {server.total_gradients}  |  Version : v{server.version}",
            ha="center", fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f2", edgecolor="#ccc"))
    plt.tight_layout()
    p = os.path.join(out_dir, "03_architecture.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    paths.append(p); print(f"  Sauvegarde : {p}")

    # ── Graphique 4 : Gradients par worker ────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = [f"Worker {w.id}" for w in workers]
    counts = [w.gradients_sent for w in workers]
    bars   = ax.bar(names, counts,
                    color=[COLORS[i % len(COLORS)] for i in range(len(workers))],
                    alpha=0.85, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Nombre de gradients envoyes")
    ax.set_title("Gradients envoyes par worker\n(independance asynchrone)",
                 fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    p = os.path.join(out_dir, "04_gradients_par_worker.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    paths.append(p); print(f"  Sauvegarde : {p}")

    return paths


# ══════════════════════════════════════════════════════════
#  RAPPORT FINAL
# ══════════════════════════════════════════════════════════

def print_report(server, workers, W_final, W_true, X, y):
    """Affiche le rapport final de l'entrainement."""
    W_err    = np.mean(np.abs(W_final - W_true))
    pred     = 1 / (1 + np.exp(-(X @ W_final)))
    accuracy = np.mean((pred > 0.5) == y)

    print("\n" + "="*60)
    print("  RAPPORT FINAL")
    print("="*60)
    print(f"  Precision finale      : {accuracy*100:.1f}%")
    print(f"  Erreur moyenne |DW|   : {W_err:.4f}")
    print(f"  Version du modele     : v{server.version}")
    print(f"  Total gradients       : {server.total_gradients}")
    for w in workers:
        print(f"  Worker {w.id}             : {w.gradients_sent} gradients")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Entrainement
    server, workers, W_final, W_true, X, y = train(
        n_workers=3,
        n_epochs=15,
        batch_size=32,
        lr=0.05
    )

    # Rapport
    print_report(server, workers, W_final, W_true, X, y)

    # Graphiques
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultats")
    print("Generation des graphiques...")
    plot_all(server, workers, W_final, W_true, out_dir=out)
    print("\nDone ! Graphiques dans le dossier resultats/")
