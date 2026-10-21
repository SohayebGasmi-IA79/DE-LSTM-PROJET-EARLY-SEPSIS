# Prédiction précoce du sepsis : LSTM + Évolution Différentielle (DE)

**Auteures :** Sohayeb Gasmi · Ikram El Ghidouni · Chaymae Al Aissaoui

## Idée principale
Prédire, **heure par heure**, si un patient en soins intensifs va entrer en sepsis (au moins ~6 h à l'avance), avec un **LSTM** (PyTorch) dont les **hyperparamètres sont optimisés par Évolution Différentielle (DE)**, implémentée entièrement de zéro.

**Question de recherche :** DE donne-t-il de meilleurs résultats qu'une configuration manuelle et qu'une recherche aléatoire (Random Search) avec le même budget de calcul ?

## Données
- PhysioNet / Computing in Cardiology Challenge 2019 (gratuites, sans inscription).
- ~40 336 patients, un fichier `.psv` par patient, une ligne par heure : 8 signes vitaux, 26 analyses de laboratoire, 6 variables statiques, `SepsisLabel`.
- Forte déséquilibre de classes (prévalence ≈ 1,7 %), d'où l'usage de l'**AUPRC** comme métrique principale.

## Pipeline
1. **EDA** : valeurs manquantes, distributions, trajectoires autour du début du sepsis.
2. **Prétraitement** : split par patient 70/15/15 (stratifié), forward-fill causal, masques de mesure, temps depuis la dernière mesure, normalisation z-score, fenêtres temporelles glissantes.
3. **Modèle** : LSTM → couche dense → probabilité de sepsis.
4. **DE** : population de configurations, mutation/croisement/sélection (variantes dont jDE), validée sur des fonctions de référence (Sphere, Rosenbrock, Rastrigin, Ackley).
5. **DE ⟷ LSTM** : minimise `1 − AUPRC_validation` sur 6 hyperparamètres.
6. **Évaluation finale** sur le test : Default vs Random Search vs DE (ROC, PR, calibration, seuil, niveau patient, bootstrap).

## Hyperparamètres optimisés
| Paramètre | Plage |
|---|---|
| hidden | 16 – 128 |
| layers | 1 – 3 |
| dropout | 0,2 – 0,6 |
| learning rate | 1e-5 – 3e-3 (log) |
| weight decay | 1e-6 – 1e-2 (log) |
| window (heures) | 6 – WMAX |

## Principes anti-fuite / équité
- Split par patient, jamais par ligne ; forward-fill uniquement (pas de backward-fill).
- Même budget d'entraînements et mêmes seeds pour Default / RS / DE.
- La sélection se fait sur la validation ; le test n'est utilisé qu'à la fin.

## Exécution
Ouvrir `sepsis_lstm_DE.ipynb` (GPU recommandé) et exécuter les cellules dans l'ordre. Dépendances : `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`.
