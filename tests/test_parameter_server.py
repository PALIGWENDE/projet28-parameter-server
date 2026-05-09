"""
Projet 28 — Tests unitaires
============================
Tests du ParameterServer et des Workers.

Auteurs : Anas ELAYATTI & Steven KABRE
"""

import sys
import os
import numpy as np
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from parameter_server import ParameterServer, Worker


# ══════════════════════════════════════════════════════════
#  TESTS PARAMETER SERVER
# ══════════════════════════════════════════════════════════

def test_init():
    """Le serveur s'initialise avec les bons attributs."""
    srv = ParameterServer(n_params=5, lr=0.1)
    assert len(srv.W)  == 5
    assert srv.lr      == 0.1
    assert srv.version == 0
    assert srv.total_gradients == 0
    print("  [OK] test_init")


def test_pull_returns_copy():
    """Pull renvoie une copie — modifier la copie ne change pas W."""
    srv    = ParameterServer(n_params=3, lr=0.1)
    W, ver = srv.pull_weights()
    W[0]   = 9999.0
    assert srv.W[0] != 9999.0, "Pull doit renvoyer une copie, pas une reference"
    print("  [OK] test_pull_returns_copy")


def test_push_updates_weights():
    """Push met a jour W correctement : W_new = W_old - lr * grad."""
    srv      = ParameterServer(n_params=3, lr=0.1)
    W_before = srv.W.copy()
    gradient = np.array([1.0, 2.0, 3.0])
    srv.push_gradient(gradient, worker_id=1, worker_version=0)
    expected = W_before - 0.1 * gradient
    assert np.allclose(srv.W, expected), "Mise a jour W incorrecte"
    assert srv.version == 1
    assert srv.total_gradients == 1
    print("  [OK] test_push_updates_weights")


def test_thread_safety():
    """Plusieurs threads pushent simultanement — pas de corruption."""
    srv = ParameterServer(n_params=10, lr=0.01)
    n_threads = 20

    def push_many(worker_id):
        for _ in range(50):
            grad = np.random.randn(10)
            W, v = srv.pull_weights()
            srv.push_gradient(grad, worker_id, v)

    threads = [threading.Thread(target=push_many, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert srv.total_gradients == n_threads * 50
    assert srv.version == n_threads * 50
    assert not np.any(np.isnan(srv.W)), "W contient des NaN — corruption detectee"
    print("  [OK] test_thread_safety")


def test_staleness():
    """Le staleness est correctement calcule."""
    srv = ParameterServer(n_params=3, lr=0.01)
    # Simuler 5 mises a jour
    for _ in range(5):
        srv.push_gradient(np.zeros(3), worker_id=99, worker_version=srv.version)
    assert srv.version == 5

    # Un worker qui avait la version 2 a un staleness de 3
    _, v = srv.pull_weights()
    staleness = v - 2
    assert staleness == 3
    print("  [OK] test_staleness")


# ══════════════════════════════════════════════════════════
#  TESTS WORKER
# ══════════════════════════════════════════════════════════

def test_worker_sigmoid():
    """Sigmoid renvoie des valeurs entre 0 et 1."""
    w = Worker.__new__(Worker)
    z = np.array([-100, -1, 0, 1, 100])
    s = Worker.sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert abs(Worker.sigmoid(np.array([0]))[0] - 0.5) < 1e-6
    print("  [OK] test_worker_sigmoid")


def test_worker_gradient_shape():
    """Le gradient a la meme dimension que W."""
    srv = ParameterServer(n_params=5, lr=0.01)
    X   = np.random.randn(10, 5)
    y   = np.random.randint(0, 2, 10).astype(float)
    w   = Worker(worker_id=1, server=srv, X=X, y=y, n_epochs=1, batch_size=10)
    W, _ = srv.pull_weights()
    loss, grad = w.compute_loss_and_gradient(W, X, y)
    assert grad.shape == W.shape
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))
    print("  [OK] test_worker_gradient_shape")


def test_worker_runs_and_sends_gradients():
    """Un worker tourne et envoie bien des gradients au serveur."""
    srv = ParameterServer(n_params=5, lr=0.05)
    X   = np.random.randn(50, 5)
    y   = np.random.randint(0, 2, 50).astype(float)
    w   = Worker(worker_id=1, server=srv, X=X, y=y, n_epochs=3, batch_size=16)
    w.start()
    w.join()
    assert w.gradients_sent > 0
    assert srv.total_gradients == w.gradients_sent
    assert len(w.losses) == 3
    print("  [OK] test_worker_runs_and_sends_gradients")


def test_worker_panne():
    """Un worker s'arrete proprement quand alive=False."""
    srv = ParameterServer(n_params=5, lr=0.05)
    X   = np.random.randn(100, 5)
    y   = np.random.randint(0, 2, 100).astype(float)
    w   = Worker(worker_id=1, server=srv, X=X, y=y, n_epochs=20, batch_size=16)
    w.start()
    time.sleep(0.1)
    w.alive = False   # Simuler une panne
    w.join(timeout=3)
    assert not w.is_alive(), "Le worker aurait du s'arreter"
    assert w.gradients_sent < 20 * (100 // 16 + 1), "Le worker aurait du s'arreter tot"
    print("  [OK] test_worker_panne")


def test_multiple_workers_converge():
    """3 workers asynchrones font converger la perte."""
    np.random.seed(0)
    W_true = np.array([1.0, -1.0, 0.5, -0.5, 0.8])
    X      = np.random.randn(300, 5)
    z      = X @ W_true
    y      = (z > 0).astype(float)
    X      = (X - X.mean(0)) / (X.std(0) + 1e-8)

    srv     = ParameterServer(n_params=5, lr=0.1)
    workers = [
        Worker(worker_id=i+1, server=srv,
               X=X[i*100:(i+1)*100], y=y[i*100:(i+1)*100],
               n_epochs=20, batch_size=16)
        for i in range(3)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    W_final, _ = srv.pull_weights()
    pred       = 1 / (1 + np.exp(-(X @ W_final)))
    accuracy   = np.mean((pred > 0.5) == y)
    assert accuracy > 0.85, f"Precision trop faible : {accuracy:.2f}"
    print(f"  [OK] test_multiple_workers_converge (precision={accuracy*100:.1f}%)")


# ══════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PROJET 28 — TESTS UNITAIRES")
    print("="*55)

    print("\nTests ParameterServer :")
    test_init()
    test_pull_returns_copy()
    test_push_updates_weights()
    test_thread_safety()
    test_staleness()

    print("\nTests Worker :")
    test_worker_sigmoid()
    test_worker_gradient_shape()
    test_worker_runs_and_sends_gradients()
    test_worker_panne()
    test_multiple_workers_converge()

    print("\n" + "="*55)
    print("  TOUS LES TESTS PASSES !")
    print("="*55 + "\n")
