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
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Lancer l'entraînement
python src/train.py

# Simuler des pannes
python src/fault_simulation.py
```

## Paramètres configurables

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `n_workers` | 3 | Nombre de workers parallèles |
| `n_epochs` | 15 | Nombre d'époques |
| `batch_size` | 32 | Taille du mini-batch |
| `lr` | 0.05 | Taux d'apprentissage (alpha) |

## Résultats

| Métrique | Valeur |
|----------|--------|
| Précision finale | 94.5% |
| Version du modèle | v720 |
| Gradients par worker | 240 |
| Temps d'entraînement | ~0.9s |

## Concepts clés

- **Asynchronisme (ASP)** : aucun worker n'attend les autres
- **Staleness** : décalage entre version locale et version serveur
- **Thread-safety** : `threading.Lock()` protège les mises à jour
- **Théorème CAP** : système AP — Disponibilité + Tolérance aux partitions

## Réalisé par

| Nom | Établissement |
|-----|---------------|
| **Anas ELAYATTI** | EMSI Marrakech — IIR S7 |
| **Steven KABRE** | EMSI Marrakech — IIR S7 |
