# ══════════════════════════════════════════════════════════
#  Projet 28 — Parameter Server
#  Dockerfile
#
#  Auteurs : Anas ELAYATTI & Steven KABRE
#  EMSI Marrakech — IIR S7 — 2024/2025
# ══════════════════════════════════════════════════════════

# Image de base Python officielle (legere)
FROM python:3.11-slim

# Metadonnees
LABEL maintainer="Anas ELAYATTI & Steven KABRE"
LABEL description="Projet 28 — Parameter Server — EMSI Marrakech IIR S7"
LABEL version="1.0"

# Repertoire de travail dans le container
WORKDIR /app

# Copier les dependances en premier (optimise le cache Docker)
COPY requirements.txt .

# Installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Creer le dossier resultats si il n'existe pas
RUN mkdir -p resultats

# Port expose (pas utilise ici mais bonne pratique)
EXPOSE 8000

# Commande par defaut : lancer l'entrainement
CMD ["python", "src/train.py"]
