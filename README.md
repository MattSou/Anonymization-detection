# Détection d'anonymisation
Résultat d'un stage de recherche de 5 mois au sein de l'Institut National de l'Audiovisuel sous la tutelle de Nicolas Hervé, sur la détection d'anonymisation au sein de vidéos via des approches de Deep Learning.

## Contenu du repo
Le rapport `Rapport_Détection_d'anonymisation.pdf` fournit l'ensemble du contexte nécessaire et l'analyse des résultats obtenus.

Le repo contient également l'ensemble des ressources nécessaires à : 
- la reproduction des principaux résultats obtenus sur la détection d'anonynimisation (dossiers [baselines](baselines) et [segmentation](segmentation))
- la création de datasets d'anonymisation, synthétiques ou non (dossier [création de dataset](./création%20de%20dataset/)) en incluant l'interface d'annotation développée pendant le stage ([interface d'annotation](https://github.com/MattSou/django_annotation_UI))
- les autres pistes et expérimentations préalables concernant le sujet (dossier [autres expérimentations](./autres%20expérimentations/))

## Set up
L'ensemble des travaux a été réalisé dans le même environnement [pyenv](https://github.com/pyenv/pyenvhttps://github.com/pyenv/pyenv), sous Python 3.9.18.
```bash
pyenv virtualenv 3.9.18 anonymization
```

Les travaux ont été réalisés avec PyTorch 1.11 et torchvision 0.12, avec CUDA 11.5.

```bash
pyenv activate anonymization

pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
```

Le reste des packages nécessaires peut être installé avec `requirements.txt`.
```bash
pyenv activate anonymization

pip install -r requirements.txt`
```



