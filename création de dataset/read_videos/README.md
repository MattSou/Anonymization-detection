## Échantillonage d'une vidéo en images
### Usage

Pour lire **une vidéo** :
```bash
python read_video.py --video_path [VIDEO_PATH] --output_dir [OUTPUT_DIR] --ips [IPS]
```
`[VIDEO_PATH]`: chemin vers la vidéo considérée

`[OUTPUT_DIR]`: dossier où créer les images

`[IPS]`: nombre d'images à échantilloner par seconde


Les images seront nommées comme il suit : 
`[VIDEO_NAME]_s_[SECONDE]_f[FRAME].jpg` où `[VIDEO_NAME]` est le dernier dossier du chemin `[OUTPUT_DIR]`.


Pour lire **toutes les vidéos** dans un dossier constitué **uniquement** de vidéos:
```bash
python read_directory.py --dir_path [DIR_PATH] --output_dir [OUTPUT_DIR] --ips [IPS]
```
`[VIDEO_PATH]`: dossier contenant les vidéos

`[OUTPUT_DIR]`: dossier où créer les images

`[IPS]`: nombre d'images à échantilloner par seconde

Les images seront stockées comme il suit :
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
