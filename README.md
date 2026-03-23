# Projet 5OD14 — Régression à noyau coopérative

Implémentation Python du projet *Cooperative Kernel Regression* (cours 5OD14) : régression ridge à noyau gaussien avec approximation de Nyström, optimisation **centralisée** (référence \(\alpha^\star\)), puis algorithmes **distribués** (réseau d’agents), **fédérés** (FedAvg, SCAFFOLD) et **DP** (DGD avec bruit laplacien).

---

## Sommaire

1. [Prérequis](#prérequis)
2. [Lancer les expériences](#lancer-les-expériences)
3. [Données](#données)
4. [Organisation du code](#organisation-du-code)
5. [Partie I — optimisation distribuée](#partie-i--optimisation-distribuée)
6. [Partie II — apprentissage fédéré](#partie-ii--apprentissage-fédéré)
7. [Partie III — confidentialité différentielle](#partie-iii--confidentialité-différentielle)
8. [Extensions Part I (robustesse)](#extensions-part-i-robustesse)
9. [Rapport PDF](#rapport-pdf)
10. [Pièges connus & réglages](#pièges-connus--réglages)
11. [Liste des figures générées](#liste-des-figures-générées)

---

## Prérequis

| Dépendance | Usage |
|------------|--------|
| **Python 3** | exécution des scripts |
| **NumPy** | matrices noyau, algorithmes |
| **Matplotlib** | figures PDF |
| **pickle** (stdlib) | chargement optionnel des bases |

Installation minimale :

```bash
pip install numpy matplotlib
```

---

## Lancer les expériences

Toutes les figures PDF sont écrites dans le dossier **`figures/`** (créé automatiquement à côté des scripts).

### Tout régénérer (recommandé pour le rendu)

```bash
cd Cooperative_learning
python3 LastName1_LastName2_LastName3_Final.py
```

Ordre d’exécution :

1. `main_part1` — Part I (algorithmes classiques + ADMM)
2. `main_part1_robustness` — graphes dirigés, pertes, async, push-sum, scaling \(n\)
3. `main_part2` — FedAvg, SCAFFOLD, participation partielle
4. `main_part3` — DGD-DP

### Lancer une partie seulement

```bash
python3 main_part1.py
python3 main_part1_robustness.py
python3 main_part2.py
python3 main_part3.py
```

---

## Données

| Fichier | Rôle |
|---------|------|
| **`first_database.pkl`** | Part I & III : \((x,y)\) global (vecteurs 1D). Si absent : génération synthétique (grande taille possible). |
| **`second_database.pkl`** | Part II : liste de shards par client `X[i], Y[i]` (souvent **non IID** entre clients). Si absent : `make_synthetic_second_database` (5 agents × 20 points). |

Les scripts utilisent les mêmes notations que le cours : noyau \(k(x,x')=\exp(-\|x-x'\|^2)\), matrices \(K_{mm}\), \(K_{nm}\), régularisation \(\nu\), bruit \(\sigma\).

---

## Organisation du code

### Cœur du problème

| Fichier | Description |
|---------|-------------|
| **`kernel_problem.py`** | `KernelDistributedProblem` : \(f_a\), \(\nabla f_a\), `grad_f_minibatch`, `F_global_centralized_form`, `build_problem`, `build_from_second_database`. |
| **`centralized.py`** | Résolution centralisée → \(\alpha^\star\), prédiction sur une grille. |
| **`utils.py`** | `Cov` / `Cov2` (noyaux), chargement des données, `savefig_pdf`, style matplotlib. |
| **`graphs.py`** | Graphes (ligne, complet, small-world), poids Metropolis, graphes **dirigés** (anneau row/column-stochastic). |

### Part I — algorithmes

| Fichier | Algorithme |
|---------|------------|
| **`dgd.py`** | DGD |
| **`gradient_tracking.py`** | Gradient tracking |
| **`dual_decomposition.py`** | Dual decomposition pair-à-paire |
| **`admm.py`** | ADMM consensus (Jacobi) |
| **`dgd_variants.py`** | DGD dirigé, pertes de paquets, async, push-sum, boucle métriques |

### Part II — fédéré

| Fichier | Description |
|---------|-------------|
| **`fedavg.py`** | FedAvg (mini-batches, participation partielle optionnelle). |
| **`scaffold.py`** | SCAFFOLD (variates de contrôle contre le client drift). |

### Part III

| Fichier | Description |
|---------|-------------|
| **`dgd_dp.py`** | DGD + bruit laplacien sur l’état (échelle liée à \(\varepsilon\)-DP cible). |

### Scripts de figures

| Script | Sorties principales |
|--------|---------------------|
| **`main_part1.py`** | Courbes d’algorithmes, topologies, reconstruction, ADMM vs \(\beta\) |
| **`main_part1_robustness.py`** | Robustesse + scaling \(n\) |
| **`main_part2.py`** | FedAvg, SCAFFOLD, participation partielle |
| **`main_part3.py`** | DGD-DP |
| **`LastName1_LastName2_LastName3_Final.py`** | Orchestre tout le pipeline |

### Autres

| Fichier | Usage |
|---------|--------|
| **`report.tex`** | Squelette de rapport (pdflatex). |
| **`sample_figure_file.py`** | Exemple minimal de figure (référence projet). |

---

## Partie I — optimisation distribuée

**Problème type :** \(n=100\) points, \(m=10\) centres Nyström, **5 agents** (20 points chacun), graphe en **ligne** avec poids **Metropolis-Hastings**.

**Métrique :** \(\max_i \|\alpha_i^t - \alpha^\star\|\) (écart au minimiseur centralisé).

**Algorithmes comparés :** DGD, Gradient Tracking, Dual Decomposition, ADMM (`\beta` fixé à `0.05` dans la comparaison principale ; courbe supplémentaire pour plusieurs \(\beta\)).

**Visualisation :** reconstruction de la fonction apprise (agent 1 vs centralisé) sur une grille.

---

## Partie II — apprentissage fédéré

**Problème :** clients = agents du *second* jeu de données ; objectif global \(F(\alpha)=\sum_a f_a(\alpha)\) aligné sur la formulation centralisée du cours.

**Métrique :** \(F(\alpha^t) - F(\alpha^\star)\) (gap d’objectif) **vs** round serveur.

**Paramètres typiques dans les figures :**

| Paramètre | Signification | Valeurs usuelles dans les scripts |
|-----------|----------------|-----------------------------------|
| \(B\) | taille de mini-batch | 15 ou 20 |
| \(E\) | époques locales par round | 1, 5, 20, 50 selon figure |
| \(C\) | clients échantillonnés par round | 5 (tous) ou 3 (partiel) |
| `lr` | pas du client | 0.002 (souvent constant) |
| `rounds` | nombre de rounds | 2000–10000 selon figure |

**FedAvg :** moyenne des modèles locaux après \(E\) époques (mini-batches de taille \(B\)).

**SCAFFOLD :** mise à jour locale avec correction \((\nabla f_i - c_i + c)\) et mise à jour des variates \(c_i\), \(c\) (Karimireddy et al., 2020) — utile quand les **20 points par client ne sont pas IID**.

**Figures « participation partielle » :**

- **`part2_fedavg_partial.pdf`** : même \(E\), comparer **\(C=5\)** vs **\(C=3\)** (mini-batch \(B=15\)).
- **`part2_fedavg_partial_E.pdf`** : **\(C=3\)** fixé, comparer **\(E \in \{1,5,50\}\)**.

---

## Partie III — confidentialité différentielle

**Algorithme :** DGD avec \(\chi_i = \alpha_i + \zeta_i\) (Laplace), puis mélange et descente (voir `dgd_dp.py`).

**Métrique :** \(\max_i \|\alpha_i^t - \alpha^\star\|\) pour trois niveaux **\(\varepsilon \in \{0.1, 1, 10\}\)** (plus petit \(\varepsilon\) ⇒ bruit plus fort ⇒ précision souvent moindre).

**Important — stabilité numérique :**

- Le pas doit être **\(O(1/L)\)** avec \(L = \max_a \lambda_{\max}(\nabla^2 f_a)\) (ici \(L\) est grand, de l’ordre \(10^1\)–\(10^2\)).
- L’échelle de bruit **ne doit pas croître** avec l’itération \(t\) (sinon explosion artificielle des courbes).

Implémentation actuelle : pas \(\eta_t \propto (0.5/L)/\sqrt{1+t}\), échelle Laplace \(\nu_t \propto (0.15/\varepsilon)/\sqrt{1+t}\).

---

## Extensions Part I (robustesse)

Script **`main_part1_robustness.py`** :

1. **`part1_robustness.pdf`** — Comparaison sur un même problème : DGD Metropolis vs **graphe dirigé** vs **pertes de paquets** vs **async** vs **push-sum** (distance à \(\alpha^\star\) + erreur de consensus).
2. **`part1_scaling_n.pdf`** — Variation de la taille de données \(n\) avec **\(m=\lceil\sqrt{n}\rceil\)**, métriques en fin d’horizon (DGD Metropolis). Le **plafond** sur \(n\) est limité (ex. `100_000`) pour la mémoire ; à ajuster selon la machine.

**Idée théorique courte :** Metropolis (doublement stochastique) tend vers la **moyenne** ; un **\(W\)** seulement row-stochastique (dirigé) ne garantit pas la moyenne — biais ; **push-sum** (\(z=s/w\)) aide sur digraphe fortement connexe ; pertes/async dégradent le consensus fixe.

---

## Rapport PDF

1. Renseigner auteurs / noms dans **`report.tex`**.
2. Compiler :

```bash
pdflatex report.tex
```

3. Renommer le PDF selon les consignes de rendu (souvent `LastName1_LastName2_LastName3_Final.pdf`).

Les figures sont incluses depuis **`figures/*.pdf`**.

---

## Pièges connus & réglages

| Sujet | Conseil |
|-------|---------|
| **ADMM** | Formulation **consensus Jacobi** sur \(\alpha\). \(\beta\) trop grand (ex. \(\gtrsim 0.15\)) peut **diverger** numériquement. Plage stable typique : `0.02`–`0.1`. |
| **DGD-DP** | Pas \(\sim 1\) sans regarder \(L\) ⇒ divergence. Bruit \(\nu_k\) **croissant** en \(k\) ⇒ courbes absurdes (\(\sim 10^{100}\)). |
| **Gros \(n\)** | `Cov2` est vectorisé dans `utils.py` ; pour \(n\) très grand, surveiller la RAM (\(K_{nm}\) de taille \(n\times m\)). |
| **Part II sans `second_database.pkl`** | Données synthétiques plus « IID » ; pour illustrer le **non-IID** et **SCAFFOLD**, utiliser la vraie base si disponible. |

---

## Liste des figures générées

Toutes dans **`figures/`**, format **PDF**.

### Part I (`main_part1.py`)

| Fichier | Contenu |
|---------|---------|
| `part1_algorithms_line.pdf` | DGD, GT, Dual Dec, ADMM — graphe ligne |
| `part1_dgd_topologies.pdf` | DGD — ligne vs small-world vs complet |
| `part1_reconstruction.pdf` | Données + prédiction centralisée vs agent 1 (GT) |
| `part1_admm_beta.pdf` | ADMM pour plusieurs \(\beta\) |

### Part I extensions (`main_part1_robustness.py`)

| Fichier | Contenu |
|---------|---------|
| `part1_robustness.pdf` | Dirigé, pertes, async, push-sum vs Metropolis |
| `part1_scaling_n.pdf` | Scaling \(n\), \(m=\lceil\sqrt{n}\rceil\) |

### Part II (`main_part2.py`)

| Fichier | Contenu |
|---------|---------|
| `part2_fedavg_E.pdf` | FedAvg — \(E \in \{1,5,50\}\), \(B=20\) |
| `part2_fedavg_diminishing.pdf` | FedAvg — pas décroissant manuel |
| `part2_fedavg_partial.pdf` | Participation partielle \(C=5\) vs \(C=3\) |
| `part2_fedavg_partial_E.pdf` | Partiel \(C=3\), \(E \in \{1,5,50\}\) |
| `part2_scaffold.pdf` | FedAvg vs SCAFFOLD pour plusieurs \(E\) |

### Part III (`main_part3.py`)

| Fichier | Contenu |
|---------|---------|
| `part3_dgd_dp.pdf` | DGD-DP pour \(\varepsilon \in \{0.1, 1, 10\}\) |

---

## Licence / usage

Projet pédagogique M2 — adapter les noms de fichiers de rendu (`LastName1_...`) selon les consignes du cours.
