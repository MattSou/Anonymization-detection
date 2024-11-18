## Overview
On propose ici d'utiliser l'architecture [SegNet](https://arxiv.org/pdf/1511.00561) pour la segmentation du flou d'anonymisation.


## Pipeline
### Fichiers de configuration
Tout repose sur le fichier de configuration .yaml choisi. Des exemples de fichier pour l'entraînement et le test sont fournis dans le dossier [configs](configs). Il détermine tous les paramètres d'entraînement, le dataset utilisé, les endroits de sauvegardes des modèles et des métadonnées.

### Forme des annotations

Les fichiers d'annotations contient 2 colonnes sans header, la première contenant le chemin complet de l'image considérée et la deuxième le chemin complet du masque. Si l'on veut ajouter des négatifs (image non altérée synthétiquement et correspondant donc à un masque uniforme à 0), on peut le préciser dans les fichiers de configuration en précisant leur emplacement dans `dataset : base_dir` et leur proportion par rapport aux images altérées synthétiquement dans `preprocessing : neg_frac`.

### Train
```bash
python train.py --name [MODEL_NAME] --annotations_path [ANNOTATIONS_PATH] --config [CONFIG_PATH]
```
`[MODEL_NAME]`: le nom donné au modèle

`[ANNOTATIONS_PATH]`: le fichier d'annotation

`[CONFIG_PATH]`: fichier de config

### Test
```bash
python test.py --model_path [MODEL_PATH] --annotations_path [ANNOTATIONS_PATH] --config [CONFIG_PATH]
```
`[MODEL_PATH]` : le chemin du modèle utilisé

`[ANNOTATIONS_PATH]`: le fichier d'annotation

`[CONFIG_PATH]`: fichier de config
