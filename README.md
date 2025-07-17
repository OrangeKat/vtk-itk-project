# Project VTK-ITK

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