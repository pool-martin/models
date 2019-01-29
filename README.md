# TensorFlow Models

This repository contains machine learning models implemented in
[TensorFlow](https://tensorflow.org). The models are maintained by their
respective authors. To propose a model for inclusion, please submit a pull
request.

Currently, the models are compatible with TensorFlow 1.0 or later. If you are
running TensorFlow 0.12 or earlier, please
[upgrade your installation](https://www.tensorflow.org/install).


## Models
- [adversarial_crypto](adversarial_crypto): protecting communications with adversarial neural cryptography.
- [adversarial_text](adversarial_text): semi-supervised sequence learning with adversarial training.
- [attention_ocr](attention_ocr): a model for real-world image text extraction.
- [autoencoder](autoencoder): various autoencoders.
- [cognitive_mapping_and_planning](cognitive_mapping_and_planning): implementation of a spatial memory based mapping and planning architecture for visual navigation.
- [compression](compression): compressing and decompressing images using a pre-trained Residual GRU network.
- [differential_privacy](differential_privacy): privacy-preserving student models from multiple teachers.
- [domain_adaptation](domain_adaptation): domain separation networks.
- [im2txt](im2txt): image-to-text neural network for image captioning.
- [inception](inception): deep convolutional networks for computer vision.
- [learning_to_remember_rare_events](learning_to_remember_rare_events):  a large-scale life-long memory module for use in deep learning.
- [lfads](lfads): sequential variational autoencoder for analyzing neuroscience data.
- [lm_1b](lm_1b): language modeling on the one billion word benchmark.
- [namignizer](namignizer): recognize and generate names.
- [neural_gpu](neural_gpu): highly parallel neural computer.
- [neural_programmer](neural_programmer): neural network augmented with logic and mathematic operations.
- [next_frame_prediction](next_frame_prediction): probabilistic future frame synthesis via cross convolutional networks.
- [object_detection](object_detection): localizing and identifying multiple objects in a single image.
- [real_nvp](real_nvp): density estimation using real-valued non-volume preserving (real NVP) transformations.
- [resnet](resnet): deep and wide residual networks.
- [skip_thoughts](skip_thoughts): recurrent neural network sentence-to-vector encoder.
- [slim](slim): image classification models in TF-Slim.
- [street](street): identify the name of a street (in France) from an image using a Deep RNN.
- [swivel](swivel): the Swivel algorithm for generating word embeddings.
- [syntaxnet](syntaxnet): neural models of natural language syntax.
- [textsum](textsum): sequence-to-sequence with attention model for text summarization.
- [transformer](transformer): spatial transformer network, which allows the spatial manipulation of data within the network.
- [tutorials](tutorials): models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).
- [video_prediction](video_prediction): predicting future video frames with neural advection.





after clone

~/docker-compose up -d --build --scale models=3 models


python scripts/tf/create_split_2kporn.py -s s1

 python scripts/tf/check_videos_limits.py 

python scripts/tf/create_sets.py --sample-rate 1 --snippet-length 1 --snippet-width 1 --engine-type opencv --split-number s1

python scripts/tf/extract_frames.py -s s1_a

python scripts/tf/convert_porn2k.py -gpu 1 -s s1_a

python slim/train_image_classifier.py --train_dir=/Exp/2kporn/art/inception_v4/s1_a/finetune/checkpoints --dataset_dir=/DL/2kporn/tfrecords/s1_a --dataset_name=porn2k     --dataset_split_name=train     --model_name=inception_v4     --checkpoint_path=/DL/initial_weigths/inception_v4/rgb_imagenet/model.ckpt --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits --save_interval_secs=3600     --optimizer=rmsprop     --normalize_per_image=1  --train_image_size=224   --max_number_of_steps=217224 --experiment_tag="Experiment: Finetune; Model: Inceptionv4; Normalization: mode 1" --experiment_file=experiment.meta --batch_size=48 --gpu_to_use=0

s1_a:
len train = 208535
len validation = 41010
len test = 257252 
steps epoch  = 208535 / batch = 4345
total steps = 217250
tempo ~ 0.925 / step 
tempo/epoca = 79 minutos


s1_b:
len train = 217470
len validation = 39261
len test = 250062 
steps epoch  = 217470 / batch = 4531
epocas = 50
total steps = 226531 
tempo ~ 0.925 / step 
tempo/epoca = 79 minutos

s2_a
len train = 219093
len validation = 38109
len test = 249611
steps epoch  = 219093 / batch = 4565
epocas = 50
total steps = 228222
tempo ~ 0.925 / step 
tempo/epoca = 79 minutos


s2_b
len train = 213528
len validation = 35546
len test = 257703
steps epoch  = 213528 / batch = 4449
epocas = 50
total steps = v4: 222425 v1: 71176
tempo ~ 0.925 / step 
tempo/epoca = 79 minutos
 
s3_a
len train = 219381
len validation = 37613
len test = 249785
steps epoch  = 219381 / batch = 4570
epocas = 50
total steps = v4: 228521 v1: 73127

s3_b
len train = 212538
len validation = 36744
len test = 257529
steps epoch  = 212538 / batch = 4428
epocas = 50
total steps = v4: 221394 v1: 70846


python slim/eval_image_classifier.py  --base_dir=/Exp/2kporn/art/inception_v4/s1_a/finetune --dataset_dir=/DL/2kporn/tfrecords/s1_a --dataset_name=porn2k     --dataset_split_name=validation --model_name=inception_v4 --eval_image_size=224 --batch_size=10 --gpu_to_use=0 


python slim/predict_image_classifier.py --alsologtostderr --base_dir=/Exp/2kporn/art/inception_v4/s1_a/imagenet_extract --checkpoint_path=/DL/initial_weigths/inception_v4/rgb_imagenet/model.ckpt --dataset_dir=/DL/2kporn/tfrecords/s1_a --task_name=label --dataset_name=porn2k --model_name=inception_v4 --preprocessing_name=porn2k --id_field_name=id --eval_replicas=1 --eval_image_size=224 --pool_features=none --pool_scores=none --extract_features --inception_layer=Mixed_7c --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits --add_scores_to_features=probs --output_format=pickle --normalize_per_image=1 --batch_size=160 --gpu_to_use=1


python slim/train_svm_layer.py --input_training /Exp/2kporn/art/inception_v1/s2_b/imagenet_extract/svm.features --output_model /Exp/2kporn/art/inception_v1/s2_b/imagenet_extract/svm.models/svm.model --jobs 5 --svm_method LINEAR_DUAL --preprocess NONE --max_iter_hyper 13


python slim/predict_svm_layer.py --input_model /Exp/2kporn/art/inception_v4/s1_a/imagenet_extract/svm.models/svm.model --input_test /Exp/2kporn/art/inception_v4/s1_a/imagenet_extract/svm.features/feats.test --pool_by_id none  --output_predictions /Exp/2kporn/art/inception_v4/s1_a/imagenet_extract/svm.predictions/test.prediction.txt --output_metrics /Exp/2kporn/art/inception_v4/s1_a/imagenet_extract/svm.predictions/test.metrics.txt --output_images /Exp/2kporn/art/inception_v4/s1_a/imagenet_extract/svm.predictions/test.images --compute_rolling_window

python scripts/results_2_etf.py --output_predictions /Exp/2kporn/art/inception_v4/s2_a/imagenet_extract/svm.predictions/test.prediction.txt --output_path /Exp/2kporn/art/inception_v4/s2_a/imagenet_extract/etf --fps_sampled 1 --set_to_process test --column k_prob_g5