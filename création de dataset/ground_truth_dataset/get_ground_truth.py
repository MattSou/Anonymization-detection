import json
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dump", type=str, required=True, help="path to the JSON dump file from the annotation UI")
parser.add_argument("--output", type=str, required=True, help="path to the output directory")

args = parser.parse_args()


def convert_timecode(timecode):
    timecode = timecode.split(':')
    return int(timecode[0])*3600 + int(timecode[1])*60 + int(timecode[2].split('.')[0])

with open(args.dump, "r") as f:
    data = json.load(f)

data = pd.json_normalize(data, max_level=0)

videos = (
    data
    .query("model == 'annotationv1.video'")
    .reset_index(drop=True)
)
videos = (
    videos
    .drop('fields', axis=1)
    .merge(pd.json_normalize(videos['fields']), left_index=True, right_index=True)
    .query("name != 'tf1_20100523T185643'")
)

keyframes = (
    data
    .query("model == 'annotationv1.keyframe'")
    .reset_index(drop=True)
)
keyframes = (
    keyframes
    .drop('fields', axis=1)
    .merge(pd.json_normalize(keyframes['fields']), left_index=True, right_index=True)
)

duration = (
    keyframes.assign(video=keyframes["video"].apply(lambda x: x.split('/')[1]))
    .groupby("video")
    .agg(duration=("timecode", "max"))
)

metadata_videos = (
    videos.assign(
        channel = videos["name"].apply(lambda x: x[:3]),
        year = videos["name"].apply(lambda x: int(x[4:][:4])),
        duration = videos["name"].apply(lambda x: duration["duration"][x]).apply(convert_timecode),
    )[["pk", "annotated.Anonym", "channel", "year", "duration"]]
    .rename(columns={"pk": "video_id"})
)
x = metadata_videos["annotated.Anonym"].value_counts()[True]
print(f'{x} videos are annotated.')


metadata_keyframes = (
    keyframes.assign(
        channel = keyframes["name"].apply(lambda x: x[:3]),
        year = keyframes["name"].apply(lambda x: int(x[4:][:4])),
    )[["pk", "video", "channel", "year", "timecode"]]
    .rename(columns={"pk": "keyframe_id", "video": "video_id"})
)
_vid = metadata_videos[metadata_videos["annotated.Anonym"] == True]["video_id"]
_x = metadata_keyframes[metadata_keyframes["video_id"].isin(_vid)]
print(f'{_x.shape[0]} keyframes are annotated.')



annotated_keyframes = keyframes[keyframes["annotated.Anonym"]==True].reset_index(drop=True)
annotated_keyframes["cat"] = annotated_keyframes.category.apply(lambda x: 1 if len(x)>0 else 0)
annotated_keyframes["name"] = annotated_keyframes["name"].apply(lambda x: x[:19]+'/'+x+'.jpg')


metadata_keyframes.to_csv(os.path.join(args.output,"metadata_keyframes.csv"), index=False)
metadata_videos.to_csv(os.path.join(args.output,"metadata_videos.csv"), index=False)
annotated_keyframes[['name', 'cat']].to_csv(os.path.join(args.output,"annotations.csv"), index=False, header=False)