## Overview
On propose ici d'utiliser la baseline composée de DINOv2 avec un classifieur SVM en sortie pour la tâche de classification binaire suivante : "l'image contient ou non du flou d'naonymisation". Elle a été utilisée principalement sur le dataset de vérités terrains annoté manuellement.


# Pipeline
## Fichiers de configuration
Tout repose sur le fichier de configuration .yaml choisi. Des exemples de fichier pour le calcul des embeddings, le fit du classifieur et le test sont fournis dans le dossier configs. Il détermine tous les paramètres d'entraînement, le dataset utilisé, les endroits de sauvegardes des modèles et des métadonnées.

## Forme des données
On attend un dataset d'images issues de vidéos organisé comme il suit : 
```
├── true_anonymized
│   ├── art_20090110T203553
│   │   ├── art_20090110T203553_s001_f0.jpg
│   │   └── ...
│   ├── art_20090220T195733
│   │   ├── art_20090220T195733_s001_f0.jpg
│   │   └── ...
│   ├── ...
│   ├── ...
│   ├── annotations.csv
│   ├── clusters.json
```

Le fichier `annotations.csv` contient 2 colonnes sans header, la première contenant le chemin de la forme `art_20090110T203553/art_20090110T203553_s001_f0.jpg` et la seconde le label `0-1`, avec une ligne par image.

Les fichiers de type `clusters.json` sont utilisés pour la déduplication du dataset et indiquent quelles images conserver pour chaque vidéo. Ils sont de la forme :
```
{
  'art_20090110T203553':[
    '[PATH TO DATASETS]/true_anonymized/art_20090110T203553/art_20090110T203553_s001_f0.jpg',
    '[PATH TO DATASETS]/true_anonymized/art_20090110T203553/art_20090110T203553_s004_f0.jpg',
    ...
  ],
  ...
  '[VIDEO]' :[
    '[PATH TO DATASETS]/[DATASET]/[VIDEO]/[IMAGE].jpg'
  ]
  ...
}
```


## Calcul des embeddings
```bash
python compute_embeddings.py --output [JSON_OUTPUT] --config [CONFIG_PATH]
```
`[JSON_OUTPUT]`: le nom donné aux JSON contenant les embeddings calculés par DINOv2

`[CONFIG_PATH]`: fichier de config

## Fit du classifieur
```bash
python fit.py --embeddings [EMBEDDINGS_PATH] --config [CONFIG_PATH]
```
`[EMBEDDINGS_PATH]`: chemin des embeddings à utiliser pour l'entraînement

`[CONFIG_PATH]`: fichier de config

## Test
```bash
python test.py --mdoel_path [SVM_PATH] --config [CONFIG_PATH]
```
`[SVM_PATH]`: chemin du fichier pickle du classifieur SVM à utiliser

`[CONFIG_PATH]`: fichier de config

