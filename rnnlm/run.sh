#!/bin/bash
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Train B without feature"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_feat.py -data B

echo "Train B+S without feature"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_feat.py -data B+S

echo "Train B+S with pos feature"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_feat.py -data B+S -feat pos

