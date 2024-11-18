## Extraction des patches flous d'une image

On performe un SLIC (Simple Linear Iterative Clustering) pour obtenir une carte de superpixel classés flous ou non, puis on utilise un algorithme glouton pour produire des patches carrés maximisant le nombre de superpixels flous en leur seins.

On peut agir dans `patch_extracion.py` sur la valeur des paramètres, notamment des seuils, pour performer la segmentation et l'extraction.
