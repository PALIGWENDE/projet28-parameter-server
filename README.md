# Projet 28 — Serveur de Paramètres (Parameter Server)

> Architecture distribuée asynchrone pour l'entraînement de modèles ML  
> EMSI Marrakech — Ingénierie Informatique et Réseaux — S7

---

## Description

Le **Parameter Server** est une architecture distribuée où :
- Un **serveur central** garde les poids `W` du modèle
- Plusieurs **workers** calculent des gradients en parallèle sur leurs données
- Les workers **pushent** leurs gradients au serveur qui met à jour `W`
- L'entraînement est **asynchrone** : chaque worker avance à son propre rythme

## Structure du projet

```
parameter-server/
├── src/
│   ├── parameter_server.py   # Classes ParameterServer et Worker
│   ├── train.py              # Entraînement distribué + visualisations
│   └── fault_simulation.py   # Simulation de pannes
├── docs/
│   ├── Rapport_Projet28.docx
│   └── CahierCharges_Projet28.pdf
├── resultats/                # Graphiques générés automatiquement
├── tests/
│   └── test_parameter_server.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Lancement avec Docker (recommandé)

### Prérequis
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé

### Commandes

```bash
# 1. Construire l'image Docker
docker build -t parameter-server .

# 2. Lancer l'entraînement
docker run --rm -v $(pwd)/resultats:/app/resultats parameter-server

# 3. Lancer les tests
docker run --rm parameter-server python tests/test_parameter_server.py

# 4. Lancer la simulation de pannes
docker run --rm -v $(pwd)/resultats:/app/resultats parameter-server python src/fault_simulation.py
```

### Avec Docker Compose (encore plus simple)

```bash
# Entraînement
docker-compose run train

# Tests
docker-compose run tests

# Simulation de pannes
docker-compose run fault
```

---

## Lancement sans Docker

```bash
pip install -r requirements.txt
python src/train.py
python tests/test_parameter_server.py
python src/fault_simulation.py
```

---

## Résultats

| Métrique | Valeur |
|----------|--------|
| Précision finale | 94.4% |
| Version du modèle | v720 |
| Gradients par worker | 240 |
| Temps d'entraînement | ~0.9s |

---

## Réalisé par

| Nom | Établissement |
|-----|---------------|
| **Anas ELAYATTI** | EMSI Marrakech — IIR S7 |
| **Steven KABRE** | EMSI Marrakech — IIR S7 |
