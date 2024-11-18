## Overview
On propose ici d'utiliser l'architecture [SegNet](https://arxiv.org/pdf/1511.00561) pour la segmentation du flou d'anonymisation.

## Création de dataset synthétique de patches flous
Le notebook [Creat_Synthetic_Patch_Dataset.ipynb](Creat_Synthetic_Patch_Dataset.ipynb) permet la création d'un dataset de patches flous à partir d'un dataset quelconque d'images nettes. Quelques paramètres sont à changer en fonction du dataset de de son organisation. Il est prévu que ce dataset soit équilibré en termes de classes de flou.

Les détails des paramètres utilisés pour le floutage sont précisés dans `dataset.py`.


## Pipeline d'entraînement/test
### Fichiers de configuration
Tout repose sur le fichier de configuration .yaml choisi. Des exemples de fichier pour l'entraînement et le test sont fournis dans le dossier [configs](configs). Il détermine tous les paramètres d'entraînement, le dataset utilisé, les endroits de sauvegardes des modèles et des métadonnées.

### Forme des annotations

Les fichiers d'annotations contient 2 colonnes sans header, la première contenant le chemin complet du patch considérée et la deuxième le label de la classe de flou (ici : `{'clear': 0, 'motion': 1, 'defocus': 2, 'gaussian': 3, 'anonym':4}`)

### Train
```bash
python train.py --name [MODEL_NAME] --config [CONFIG_PATH]
```
`[MODEL_NAME]`: le nom donné au modèle

`[CONFIG_PATH]`: fichier de config

### Test
```bash
python test.py --model_path [MODEL_PATH] --config [CONFIG_PATH]
```
`[MODEL_PATH]` : le chemin du modèle utilisé

`[CONFIG_PATH]`: fichier de config
