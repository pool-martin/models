# Tutorial Models

This folder contains models referenced to from the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).


after clone

~/docker-compose up -d --build --scale models=3 models


python scripts/tf/create_split_2kporn.py -s s1

 python scripts/tf/check_videos_limits.py 

python scripts/tf/create_sets.py --sample-rate 1 --snippet-length 1 --snippet-width 1 --engine-type opencv --split-number s1

python scripts/tf/extract_frames.py -s s1_a