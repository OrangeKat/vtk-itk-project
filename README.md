# Project VTK-ITK

## Auteurs
- Gabriel Cellier
- Thomas POLO
- Leopold TRAN


## Dependencies
- itk>=5.3.0
- vtk>=9.2.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- nbformat>=5.0.0
- nbconvert>=6.0.0

## How to run

```
pip install -r requirements.txt
python3 main.py
```

## Évaluation et sélection des méthodes de recalage

Dans le cadre de cette étude, plusieurs méthodes de recalage d’images ont été comparées afin d’identifier la technique la plus adaptée en termes de précision et de robustesse. Les approches testées incluent le recalage rigide, affine et par translation. Chaque méthode a été évaluée quantitativement à l’aide de deux métriques : l’erreur quadratique moyenne (Mean Squared Error, MSE) et la corrélation croisée normalisée (Normalized Cross-Correlation, NCC).

Les résultats sont les suivants :

    Recalage rigide : MSE = 27126,7 ; NCC = 0,641

    Recalage affine : MSE = 24436,4 ; NCC = 0,682

    Recalage par translation : MSE = 15762,3 ; NCC = 0,802

L’analyse comparative de ces valeurs montre une amélioration significative de la qualité de recalage avec la méthode par translation. Cette dernière présente à la fois la plus faible erreur quadratique moyenne et la plus forte corrélation, indiquant une meilleure superposition entre l’image source et l’image cible.

Cette exploration systématique des méthodes et paramètres de recalage a permis de justifier rigoureusement le choix final de la méthode utilisée. Le recalage par translation a ainsi été retenu pour les traitements ultérieurs, en raison de sa performance supérieure sur les jeux de données étudiés.



## Évaluation de la visualisation 

1. Images d'origine (case6_gre1.nrrd et case6_gre2.nrrd)
J'ai utilisé ces deux images IRM du même patient prises à des moments différents parce que :

Comparaison temporelle : Pour voir comment la tumeur évolue dans le temps, il faut au moins deux acquisitions
Même modalité : Les deux sont des IRM GRE, donc on peut les comparer directement
Bonne qualité : Les tumeurs sont bien visibles (zones blanches/claires) sur fond de cerveau gris
Format NRRD : Compatible avec ITK, permet l'analyse 3D complète

2. Image recalée (translation.nrrd)
J'ai fait un recalage parce que :

Mouvement du patient : Entre les deux scans, le patient a bougé légèrement
Comparaison précise : Sans recalage, on ne peut pas comparer pixel par pixel
Algorithme de translation : J'ai utilisé une translation simple car les images étaient déjà bien alignées
Validation : Le déplacement de seulement 3.32mm montre que le recalage a bien marché

3. Segmentations (masques blancs)
Les petites zones blanches dans mes résultats représentent uniquement les tumeurs parce que :

Seuillage connecté : J'ai utilisé des seuils [500-800] pour isoler les tissus tumoraux hyperintenses
Point de départ : Le seed point (90,70,51) était placé manuellement dans la tumeur
Résultat propre : Contrairement à une segmentation globale, ça évite de prendre tout le cerveau
Validation visuelle : Les masques correspondent bien aux zones tumorales visibles sur les images originales

4. Visualisation comparative 2D
J'ai créé cette vue avec 8 panneaux pour montrer :

Images originales : Pour voir les données de départ
Overlays colorés : Rouge pour T1, bleu pour T2, pour voir les différences de forme
Carte de différence : Les zones rouges/bleues montrent où l'intensité a changé
Graphique de volume : Visualisation simple du changement (+6.2%)
Résumé des métriques : Toutes les mesures importantes dans un coin

6. Métriques calculées
Volume (+6.2%) : Simple à comprendre, utilisé en clinique pour suivre l'évolution
Intensité (-4.4%) : Peut indiquer une réponse au traitement ou un changement de nature tumorale
Déplacement (3.32mm) : Faible = bon recalage, sinon ça indiquerait un problème technique
Recouvrement (72%) : Mesure la similarité spatiale, 72% c'est bien pour des tumeurs qui évoluent