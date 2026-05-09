"""
Projet 28 — Simulation de Pannes
=================================
Teste la robustesse de l'architecture Parameter Server
face a differents scenarios de pannes.

Scenarios :
    1. Panne d'un worker en cours d'entrainement
    2. Worker lent (staleness eleve)
    3. Comparaison normal vs panne

Auteurs : Anas ELAYATTI & Steven KABRE
EMSI Marrakech — IIR S7 — 2024/2025
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from parameter_server import ParameterServer, Worker
from train import generate_data, split_for_workers

np.random.seed(42)

COLORS = ["#185FA5", "#BA7517", "#3B6D11", "#A32D2D"]


# ══════════════════════════════════════════════════════════
#  SCENARIO 1 — Panne d'un worker
# ══════════════════════════════════════════════════════════

def scenario_panne_worker(kill_after: float = 1.5):
    """
    Simule la panne du Worker 2 apres `kill_after` secondes.

    Le Worker 2 est tue (alive=False) pendant l'entrainement.
    Les Workers 1 et 3 continuent sans interruption.

    Args:
        kill_after (float) : secondes avant de tuer le Worker 2

    Returns:
        dict : resultats du scenario
    """
    print("\n" + "="*60)
    print("  SCENARIO 1 — Panne du Worker 2")
    print(f"  Worker 2 sera tue apres {kill_after}s")
    print("="*60)

    X, y, W_true = generate_data()
    splits       = split_for_workers(X, y, 3)
    server       = ParameterServer(n_params=X.shape[1], lr=0.05)

    workers = []
    for i, (Xi, yi) in enumerate(splits):
        w = Worker(worker_id=i+1, server=server,
                   X=Xi, y=yi, n_epochs=15, batch_size=32)
        workers.append(w)

    # Lancer tous les workers
    t0 = time.time()
    for w in workers:
        w.start()

    # Tuer le Worker 2 apres kill_after secondes
    def kill_worker2():
        time.sleep(kill_after)
        workers[1].alive = False
        print(f"\n  *** PANNE : Worker 2 tue a t={kill_after}s ***\n")

    killer = threading.Thread(target=kill_worker2, daemon=True)
    killer.start()

    for w in workers:
        w.join()
    elapsed = time.time() - t0

    # Evaluation
    W_final, _ = server.pull_weights()
    pred       = 1 / (1 + np.exp(-(X @ W_final)))
    accuracy   = np.mean((pred > 0.5) == y)

    print(f"\n  Worker 1 : {workers[0].gradients_sent} gradients envoyes")
    print(f"  Worker 2 : {workers[1].gradients_sent} gradients envoyes (PANNE)")
    print(f"  Worker 3 : {workers[2].gradients_sent} gradients envoyes")
    print(f"  Precision finale : {accuracy*100:.1f}%")
    print(f"  Temps total      : {elapsed:.2f}s")
    print(f"  Conclusion       : L'entrainement a continue malgre la panne !")

    return {
        "scenario"      : "panne_worker",
        "accuracy"      : accuracy,
        "elapsed"       : elapsed,
        "gradients"     : [w.gradients_sent for w in workers],
        "total_updates" : server.total_gradients,
        "workers"       : workers,
        "server"        : server,
    }


# ══════════════════════════════════════════════════════════
#  SCENARIO 2 — Worker lent (staleness eleve)
# ══════════════════════════════════════════════════════════

def scenario_worker_lent(slow_delay: float = 0.02):
    """
    Simule un Worker 3 lent (staleness eleve).

    Le Worker 3 a un delai artificiel `slow_delay` entre chaque
    mini-batch, simulant une machine sous-performante.

    Args:
        slow_delay (float) : delai en secondes par mini-batch

    Returns:
        dict : resultats du scenario
    """
    print("\n" + "="*60)
    print(f"  SCENARIO 2 — Worker 3 lent (delai={slow_delay}s/batch)")
    print("="*60)

    X, y, W_true = generate_data()
    splits       = split_for_workers(X, y, 3)
    server       = ParameterServer(n_params=X.shape[1], lr=0.05)

    workers = [
        Worker(worker_id=1, server=server, X=splits[0][0], y=splits[0][1],
               n_epochs=15, batch_size=32, delay=0.0),
        Worker(worker_id=2, server=server, X=splits[1][0], y=splits[1][1],
               n_epochs=15, batch_size=32, delay=0.0),
        Worker(worker_id=3, server=server, X=splits[2][0], y=splits[2][1],
               n_epochs=15, batch_size=32, delay=slow_delay),  # Worker lent
    ]

    t0 = time.time()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    elapsed = time.time() - t0

    W_final, _ = server.pull_weights()
    pred       = 1 / (1 + np.exp(-(X @ W_final)))
    accuracy   = np.mean((pred > 0.5) == y)

    print(f"\n  Worker 1 : {workers[0].gradients_sent} gradients (rapide)")
    print(f"  Worker 2 : {workers[1].gradients_sent} gradients (rapide)")
    print(f"  Worker 3 : {workers[2].gradients_sent} gradients (LENT)")
    print(f"  Precision finale : {accuracy*100:.1f}%")
    print(f"  Temps total      : {elapsed:.2f}s")
    print(f"  Conclusion       : Le staleness n'empeche pas la convergence !")

    return {
        "scenario"      : "worker_lent",
        "accuracy"      : accuracy,
        "elapsed"       : elapsed,
        "gradients"     : [w.gradients_sent for w in workers],
        "total_updates" : server.total_gradients,
        "workers"       : workers,
        "server"        : server,
    }


# ══════════════════════════════════════════════════════════
#  SCENARIO 3 — Entrainement normal (reference)
# ══════════════════════════════════════════════════════════

def scenario_normal():
    """
    Entrainement normal sans panne — sert de reference pour la comparaison.

    Returns:
        dict : resultats du scenario
    """
    print("\n" + "="*60)
    print("  SCENARIO 3 — Entrainement normal (reference)")
    print("="*60)

    X, y, W_true = generate_data()
    splits       = split_for_workers(X, y, 3)
    server       = ParameterServer(n_params=X.shape[1], lr=0.05)

    workers = [
        Worker(worker_id=i+1, server=server,
               X=splits[i][0], y=splits[i][1],
               n_epochs=15, batch_size=32)
        for i in range(3)
    ]

    t0 = time.time()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    elapsed = time.time() - t0

    W_final, _ = server.pull_weights()
    pred       = 1 / (1 + np.exp(-(X @ W_final)))
    accuracy   = np.mean((pred > 0.5) == y)

    print(f"\n  Precision finale : {accuracy*100:.1f}%")
    print(f"  Temps total      : {elapsed:.2f}s")
    print(f"  Gradients total  : {server.total_gradients}")

    return {
        "scenario"      : "normal",
        "accuracy"      : accuracy,
        "elapsed"       : elapsed,
        "gradients"     : [w.gradients_sent for w in workers],
        "total_updates" : server.total_gradients,
        "workers"       : workers,
        "server"        : server,
    }


# ══════════════════════════════════════════════════════════
#  VISUALISATION COMPARAISON
# ══════════════════════════════════════════════════════════

def plot_comparaison(results: list, out_dir: str = "resultats"):
    """
    Genere un graphique comparant les 3 scenarios.

    Args:
        results  : liste des dicts retournes par les scenarios
        out_dir  : dossier de sortie
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Projet 28 — Comparaison des scenarios de robustesse",
                 fontsize=13, fontweight="bold")

    labels    = [r["scenario"].replace("_", "\n") for r in results]
    accs      = [r["accuracy"] * 100 for r in results]
    times     = [r["elapsed"] for r in results]
    totgrads  = [r["total_updates"] for r in results]

    # Precision
    ax = axes[0]
    bars = ax.bar(labels, accs, color=COLORS[:len(results)], alpha=0.85, width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision finale par scenario")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # Temps
    ax = axes[1]
    bars = ax.bar(labels, times, color=COLORS[:len(results)], alpha=0.85, width=0.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}s", ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Temps (secondes)")
    ax.set_title("Temps d'entrainement")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # Gradients
    ax = axes[2]
    bars = ax.bar(labels, totgrads, color=COLORS[:len(results)], alpha=0.85, width=0.5)
    for bar, val in zip(bars, totgrads):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Gradients recus")
    ax.set_title("Total gradients traites")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "05_comparaison_scenarios.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Graphique sauvegarde : {p}")
    return p


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  PROJET 28 — SIMULATION DE PANNES")
    print("#"*60)

    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "resultats"
    )

    # Lancer les 3 scenarios
    r_normal = scenario_normal()
    r_panne  = scenario_panne_worker(kill_after=1.5)
    r_lent   = scenario_worker_lent(slow_delay=0.02)

    # Comparaison
    print("\n\nGeneration du graphique de comparaison...")
    plot_comparaison([r_normal, r_panne, r_lent], out_dir=out)

    # Tableau recap
    print("\n" + "="*60)
    print("  TABLEAU RECAP")
    print("="*60)
    print(f"  {'Scenario':<20} {'Precision':>10} {'Temps':>8} {'Gradients':>10}")
    print(f"  {'-'*50}")
    for r in [r_normal, r_panne, r_lent]:
        print(f"  {r['scenario']:<20} {r['accuracy']*100:>9.1f}% "
              f"{r['elapsed']:>7.2f}s {r['total_updates']:>10}")
    print("="*60)
    print("\nConclusion : L'architecture est robuste aux pannes workers !")
