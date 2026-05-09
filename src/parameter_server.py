"""
Projet 28 — Serveur de Parametres (Parameter Server)
=====================================================
Architecture distribuee asynchrone pour l'entrainement de modeles ML.

Classes :
    - ParameterServer : serveur central qui stocke et met a jour les poids W
    - Worker          : thread independant qui calcule les gradients

Auteurs : Anas ELAYATTI & Steven KABRE
EMSI Marrakech — IIR S7 — 2024/2025
"""

import threading
import numpy as np
import time
import logging
from typing import Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  SERVEUR DE PARAMETRES
# ══════════════════════════════════════════════════════════

class ParameterServer:
    """
    Serveur central de l'architecture distribuee.

    Responsabilites :
        - Stocker les poids W du modele (parametres partages)
        - Recevoir les gradients des workers (push)
        - Renvoyer les poids a jour (pull)
        - Garantir la thread-safety via un mutex (threading.Lock)

    Attributs :
        W        (np.ndarray) : vecteur des poids du modele
        lr       (float)      : taux d'apprentissage alpha
        version  (int)        : numero de version des poids
        lock     (Lock)       : verrou pour les acces concurrents
    """

    def __init__(self, n_params: int, lr: float = 0.01):
        """
        Initialise le serveur avec des poids aleatoires proches de zero.

        Args:
            n_params (int)  : dimension du vecteur de poids W
            lr       (float): taux d'apprentissage (defaut 0.01)
        """
        self.W               = np.random.randn(n_params) * 0.01
        self.lr              = lr
        self.version         = 0
        self.total_gradients = 0
        self.lock            = threading.Lock()
        self.loss_history    : List[float] = []
        self._active_workers = 0

    # ── PUSH ──────────────────────────────────────────────
    def push_gradient(self, gradient: np.ndarray,
                      worker_id: int, worker_version: int) -> None:
        """
        Recoit un gradient d'un worker et met a jour W immediatement.

        Formule : W = W - lr * gradient   (descente de gradient)

        Args:
            gradient       (np.ndarray) : vecteur gradient calcule par le worker
            worker_id      (int)        : identifiant du worker
            worker_version (int)        : version des poids utilisee pour le calcul
        """
        with self.lock:
            staleness = self.version - worker_version
            self.W              -= self.lr * gradient
            self.version        += 1
            self.total_gradients += 1
            log.info(
                f"  [SERVEUR] << Worker {worker_id} | "
                f"gradient recu | staleness={staleness} | v{self.version}"
            )

    # ── PULL ──────────────────────────────────────────────
    def pull_weights(self) -> Tuple[np.ndarray, int]:
        """
        Renvoie une copie des poids actuels et leur version.

        Returns:
            (np.ndarray, int) : copie de W et numero de version
        """
        with self.lock:
            return self.W.copy(), self.version

    # ── UTILITAIRES ───────────────────────────────────────
    def record_loss(self, loss: float) -> None:
        """Enregistre une valeur de perte dans l'historique."""
        with self.lock:
            self.loss_history.append(loss)

    def get_stats(self) -> dict:
        """Retourne les statistiques actuelles du serveur."""
        with self.lock:
            return {
                "version"         : self.version,
                "total_gradients" : self.total_gradients,
                "n_params"        : len(self.W),
                "lr"              : self.lr,
            }


# ══════════════════════════════════════════════════════════
#  WORKER
# ══════════════════════════════════════════════════════════

class Worker(threading.Thread):
    """
    Worker independant — herite de threading.Thread.

    Cycle de vie (boucle asynchrone) :
        1. PULL  : recupere les poids W du serveur
        2. CALCUL: calcule gradient sur son mini-batch
        3. PUSH  : envoie le gradient au serveur
        4. Repeter jusqu'a la fin des epoques

    L'asynchronisme vient du fait que les workers n'attendent
    pas les uns les autres — chacun avance a son propre rythme.

    Attributs :
        id           (int)        : identifiant unique du worker
        server       (ParameterServer) : reference au serveur central
        X            (np.ndarray) : sous-ensemble des features
        y            (np.ndarray) : sous-ensemble des labels
        n_epochs     (int)        : nombre d'epoques a effectuer
        batch_size   (int)        : taille du mini-batch
        local_version(int)        : version des poids utilisee localement
        gradients_sent(int)       : compteur de gradients envoyes
        losses       (List[float]): historique des pertes par epoque
        alive        (bool)       : False = simulation de panne
    """

    def __init__(self, worker_id: int, server: ParameterServer,
                 X: np.ndarray, y: np.ndarray,
                 n_epochs: int, batch_size: int,
                 delay: float = 0.0):
        """
        Args:
            worker_id  (int)   : identifiant du worker (1, 2, 3...)
            server     (ParameterServer) : serveur central
            X          (np.ndarray)      : donnees du worker
            y          (np.ndarray)      : labels du worker
            n_epochs   (int)   : nombre d'epoques
            batch_size (int)   : taille du mini-batch
            delay      (float) : delai artificiel en secondes (simule un worker lent)
        """
        super().__init__(daemon=True)
        self.id             = worker_id
        self.server         = server
        self.X              = X
        self.y              = y
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.delay          = delay
        self.local_version  = 0
        self.gradients_sent = 0
        self.losses         : List[float] = []
        self.alive          = True   # False = simulation de panne

    # ── MODELE ────────────────────────────────────────────
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Fonction d'activation sigmoid : 1 / (1 + exp(-z))"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))

    def compute_loss_and_gradient(
            self, W: np.ndarray,
            X_batch: np.ndarray,
            y_batch: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calcule la perte et le gradient pour un mini-batch.

        Modele : regression logistique binaire
            prediction p = sigmoid(X @ W)
            perte L      = Binary Cross-Entropy
            gradient     = (1/n) * X^T @ (p - y)

        Args:
            W       : poids actuels du modele
            X_batch : features du mini-batch  (n x d)
            y_batch : labels du mini-batch    (n,)

        Returns:
            (loss, gradient) : scalaire et vecteur de meme dim que W
        """
        n    = len(y_batch)
        z    = X_batch @ W
        pred = self.sigmoid(z)
        eps  = 1e-9
        loss = -np.mean(
            y_batch * np.log(pred + eps) +
            (1 - y_batch) * np.log(1 - pred + eps)
        )
        gradient = (1.0 / n) * X_batch.T @ (pred - y_batch)
        return loss, gradient

    # ── BOUCLE PRINCIPALE ─────────────────────────────────
    def run(self) -> None:
        """
        Boucle d'entrainement asynchrone du worker.
        Appele automatiquement par threading lors du worker.start()
        """
        n = len(self.X)
        log.info(
            f"Worker {self.id} demarre | "
            f"{n} exemples | {self.n_epochs} epoques"
        )

        for epoch in range(self.n_epochs):

            # Simulation de panne en cours d'entrainement
            if not self.alive:
                log.info(f"Worker {self.id} : PANNE simulee — arret a l'epoque {epoch+1}")
                break

            # Melanger les donnees a chaque epoque
            idx        = np.random.permutation(n)
            X_shuffled = self.X[idx]
            y_shuffled = self.y[idx]
            epoch_losses = []

            # Decouper en mini-batches et traiter
            for start in range(0, n, self.batch_size):
                if not self.alive:
                    break

                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]

                # 1. PULL — obtenir les derniers poids
                W, self.local_version = self.server.pull_weights()

                # 2. CALCUL — gradient sur le mini-batch
                loss, gradient = self.compute_loss_and_gradient(W, X_batch, y_batch)
                epoch_losses.append(loss)

                # 3. PUSH — envoyer le gradient au serveur
                self.server.push_gradient(gradient, self.id, self.local_version)
                self.gradients_sent += 1

                # Delai optionnel (simule un worker lent)
                if self.delay > 0:
                    time.sleep(self.delay)
                else:
                    time.sleep(np.random.uniform(0.001, 0.004))

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.losses.append(avg_loss)
            self.server.record_loss(avg_loss)

            log.info(
                f"Worker {self.id} | epoque {epoch+1}/{self.n_epochs} | "
                f"perte={avg_loss:.4f} | gradients={self.gradients_sent}"
            )

        log.info(
            f"Worker {self.id} TERMINE | "
            f"total gradients envoyes : {self.gradients_sent}"
        )
