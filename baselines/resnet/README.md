## Overview
On propose ici d'utiliser la baseline composée d'un ResNet50 avec un classifieur binaire en sortie pour la tâche de classification binaire suivante : "l'image contient ou non du flou d'naonymisation". Elle a été utilisée principalement sur le dataset de vérités terrains annoté manuellement.


## Pipeline
### Fichiers de configuration
Tout repose sur le fichier de configuration .yaml choisi. Des exemples de fichier pour l'entraînement et le test sont fournis dans le dossier [configs](configs). Il détermine tous les paramètres d'entraînement, le dataset utilisé, les endroits de sauvegardes des modèles et des métadonnées.

### Forme des données
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
│   ├── metadata_keyframes.csv
│   ├── metadata_videos.csv
│   ├── clusters.json
```

Le fichier `annotations.csv` contient 2 colonnes sans header, la première contenant le chemin de la forme `art_20090110T203553/art_20090110T203553_s001_f0.jpg` et la seconde le label `0-1`, avec une ligne par image.

Le fichier `metadata_keyframes.csv` contient 5 colonnes : `keyframe_id`, `video_id`, `channel`, `year` et `timecode` avec une ligne par image. Les colonnes `keyframe_id` et `video_id` sont de la forme : `true_anonymized/art_20090110T203553/art_20090110T203553_s001_f0.jpg` et `true_anonymized/art_20090110T203553`.

Le fichier `metadata_videos.csv` contient 5 colonnes : `video_id`, `annotated.Anonym`, `channel`, `year` et `duration` avec une ligne par image. La colonne `video_id` est de la forme `true_anonymized/art_20090110T203553` et la colonne `annotated.Anonym` est un booléen indiquant si la vidéo a été entièrement annotée.

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

## Train
```bash
python train.py --name [MODEL_NAME] --config [CONFIG_PATH]
```
`[MODEL_NAME]`: le nom donné au modèle

`[CONFIG_PATH]`: fichier de config

## Test
```bash
python test.py --model_path [MODEL_PATH] --config [CONFIG_PATH]
```
`[MODEL_PATH]`: chemin du modèle à tester

`[CONFIG_PATH]`: fichier de config



