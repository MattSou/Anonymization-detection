## Génération d'annotations et de metadata à partir du fichier issu de l'interface d'annotation
### Usage
```bash
python get_ground_truth.py --dump [DUMP] --output [OUTPUT_DIR]
```
`[DUMP]`: fichier JSON issu de [l'interface d'annotaion](../../django_annotation_UI/)

`[OUTPUT_DIR]`: dossier où créer les annotations et les métadonnées. Leur forme est précisée dans ce [fichier](../../baselines/resnet/README.md).

