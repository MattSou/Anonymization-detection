## Déduplication
Méthode pour dédupliquer un dataset d'images extraites de vidéos. 

On performe un clustering K-Means au sein de chaque vidéo basé sur la similarité cosinus entre les images. On obtient ainsi un dataset avec la même diversité mais en général une image représenant par plan.

L'avantage est de pouvoir ensuite entraîner un modèle beaucoup fois sur les mêmes images que peu de fois sur des beaucoup d'images très ressemblantes entre elles, et de gagner un peu de temps d'entraînement.
