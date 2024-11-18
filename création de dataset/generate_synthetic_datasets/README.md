## Génération de dataset synthétique contenant des visages floutés
### Usage
```bash
python generate_synthetic_dataset.py --base_dir [BASE_DIR] --write --target_dir [TARGET_DIR]
```
`[BASE_DIR]`: dossier contant le dataset d'images extraites de la vidéo

`[TARGET_DIR]`: dossier de destination souhaité

### Forme des données de base
On attend un dataset d'images issues de vidéos organisé comme il suit : 
```
├── [OUTPUT_DIR]
│   ├── [VIDEO_NAME]
│   │   ├── [IMAGE].jpg
│   │   └── ...
│   ├── [VIDEO_NAME]
│   │   ├── [IMAGE].jpg
│   │   └── ...
│   ├── ...
│   ├── ...
``` 

### Méthode
On applique un détecteur de visage [YOLOv5](https://github.com/elyha7/yoloface) sur chaque image, et on sélectionne le visage le plus grand et le plus "de face" dans l'image pour lui appliquer un flou rectangulaire et l'anonymiser. On produit des annotations en plus, contenant la position du visage dans l'image.

### Résultat
Le script produit un dossier d'images organisé comme il suit : 
```
├── [TARGET_DIR]
│   ├── [VIDEO_NAME]
│   │   ├── [ANONYMIZED_IMAGE].jpg
│   │   └── ...
│   ├── [VIDEO_NAME]
│   │   ├── [ANONYMIZED_IMAGE].jpg
│   │   └── ...
│   ├── ...
│   ├── ...
│   ├── annotations.json
``` 
qui ne contient que les images effectivement anonymisées (qui contiennent donc un visage satisfaisant). Le fichier `annotations.json` contient les informations récupérées durant l'anonymisation sous la forme suivante :

```python
"img.jpg": {
        "coord": {
            "x1": 113,
            "y1": 55,
            "x2": 176,
            "y2": 105
        },
        "surface%": 0.03076171875,
        "laplacian": 12.056397178130512,
        "global_laplacian": 1316.7408760146138,
        "blur_score": 1.2152848355555557,
        "front_score": 0.2952403048205942,
        "class": 0
    }
```
**Paramètres sauvegardés :** 
- `"coord"` : coordonnées du visage trouvé.
- `"surface%"` : proportion occupée par le visage dans l'image. Un visage n'est retenu que s'il occupe une proportion suffisante de l'image (suffisamment grand).
- `"laplacian"`, `"global laplacian"` et `"class"` : mesure de la netteté du visage. On veut flouter des visages nets !
- `"front_score"` : mesure d'à quel point le visage et de face (graĉe aux landmarks de YOLOv5). 
- `"class"` : si le visage à été anonymisé (1 si oui, 0 sinon). C'est le cas s'il respecte les critères précédents : taille suffaisamment grande, visage suffisamment net, visage suffisamment de face.

Les valeurs seuil sont précisées dans `generate_synthetic_dataset.py` et peuvent donc être changées.

Si aucun visage dans l'image n'est trouvé, l'annotation ressemble à cela :
```python
"img.jpg": {
        "class": 0
    }
```

## Création de masques de segmentation pour l'annotation
Après avoir généré ce dataset synthétique et à partir d'un fichier *JSON* formaté exactement comme dans l'exemple précédent, on peut également produire des masques de segmentation binaire pour des tâches comme celle décrite dans [SegNet](../../segmentation/SegNet/).
```bash
python produce_masks.py --base_dir [BASE_DIR] --anonym_dir [ANONYM_DIR] --target_dir [TARGET_DIR] --annotation_file [ANNOTATION_FILE]
```
`[BASE_DIR]`: dossier contant le dataset d'images extraites de la vidéo

`[ANONYM_DIR]`: dossier le dataset synthétique anonymisé

`[TARGET_DIR]`: dossier de destination souhaité pour les masques

`[ANNOTATION_FILE]`: dfichier JSON d'annotation produit lors de la génération du dataset synthétique

Le script produit un dossier de masques organisé comme il suit : 
```
├── [TARGET_DIR]
│   ├── [VIDEO_NAME]
│   │   ├── [MASK].jpg
│   │   └── ...
│   ├── [VIDEO_NAME]
│   │   ├── [MASK].jpg
│   │   └── ...
│   ├── ...
│   ├── ...
│   ├── mask_annotations.csv
``` 

`mask_annotations.csv` est un fichier d'annotations sans header : la première colonne contient le chemin vers les images anonymisées, la seconde le chemin vers les masques correspondants.