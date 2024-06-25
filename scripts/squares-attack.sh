#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 2"

Q=ColorA # ColorA or ColorB
T=1      # desired target color value

GPU=0
OUTPUT_PATH='/home/ducha/Workspace/tmp/ace_squares'
MODELPATH='/home/ducha/Workspace/ML/Models/square_ddpm/990000.pt'
DATAPATH='/home/ducha/Workspace/Data/peal/square_pt'
CLASSIFIERPATH="/home/ducha/Nextcloud/Uni-Master/Thesis/notebooks/squares/ace/square_logs/square_lenet/version_0/checkpoints/epoch=19-step=8000.ckpt" # TODO: Train our own classifier?

EXPNAME=squares_ace_0.2_noisefrac

python main.py $MODEL_FLAGS $SAMPLE_FLAGS \
    --gpu $GPU \
    --num_samples 10 \
    --model_path $MODELPATH \
    --classifier_path $CLASSIFIERPATH \
    --output_path $OUTPUT_PATH \
    --exp_name $EXPNAME \
    --attack_method PGD \
    --attack_iterations 50 \
    --attack_joint True \
    --dist_l1 0.001 \
    --timestep_respacing 50 \
    --sampling_time_fraction 0.2 \
    --sampling_stochastic True \
    --sampling_inpaint 0.15 \
    --label_query_str $Q \
    --label_target $T \
    --image_size 64 \
    --data_dir $DATAPATH \
    --dataset squares
