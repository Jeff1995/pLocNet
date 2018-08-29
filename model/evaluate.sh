#!/bin/bash
set -e


#===============================================================================
#
#  Preparation
#
#===============================================================================

seed=0
device=(0 1 2 3)
no_fit=""
#no_fit="-n"

# For use in parallel
rpl="{d} 1 @pool = ($(echo ${device[@]} | tr ' ' ',')); \$_ = \$pool[(\$job->slot() - 1) % (\$#pool + 1)]"


#===============================================================================
#
#  Feature learning
#
#===============================================================================

# Unsupervised CNN-VAE embedding
./run_embedding.py \
    -i ../data/preprocessed/sequence.h5 \
    -o ../result/cnnvae \
    -s ${seed} -d ${device[1]} ${no_fit}

# Supervised CNN embedding
parallel -j ${#device[@]} --rpl "${rpl}" -v ./run_cnn.py \
    -x ../data/preprocessed/sequence90.h5 \
    -y ../data/preprocessed/localization.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/cnn_fold{} \
    -s ${seed} -d {d} ${no_fit} --save-hidden \
::: $(seq 0 4)


#===============================================================================
#
#  MLP Prediction
#
#===============================================================================

# Unsupervised CNN-VAE embedding -> MLP
parallel --rpl "${rpl}" -v ./run_mlp.py \
    -x ../result/cnnvae/result.h5 \
    -y ../data/preprocessed/localization.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/cnnvae_mlp_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)

# Supervised CNN embedding -> MLP
parallel --rpl "${rpl}" -v ./run_mlp.py \
    -x ../result/cnn_fold{}/hidden.h5 \
    -y ../data/preprocessed/localization.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/cnn_mlp_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)

# kmer -> MLP
parallel --rpl "${rpl}" -v ./run_mlp.py \
    -x ../data/preprocessed/feature_3mer.h5 \
    -y ../data/preprocessed/localization.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/3mer_mlp_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)


#===============================================================================
#
#  GCN prediction
#
#===============================================================================

# Unsupervised CNN-VAE embedding -> GCN
parallel --rpl "${rpl}" -v ./run_gcn.py \
    -x ../result/cnnvae/result.h5 \
    -y ../data/preprocessed/localization.h5 \
    -g ../data/preprocessed/ppi.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/cnnvae_gcn_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)

# Supervised CNN embedding -> GCN
parallel --rpl "${rpl}" -v ./run_gcn.py \
    -x ../result/cnn_fold{}/hidden.h5 \
    -y ../data/preprocessed/localization.h5 \
    -g ../data/preprocessed/ppi.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/cnn_gcn_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)

# kmer -> GCN
parallel --rpl "${rpl}" -v ./run_gcn.py \
    -x ../data/preprocessed/feature_3mer.h5 \
    -y ../data/preprocessed/localization.h5 \
    -g ../data/preprocessed/ppi.h5 \
    --split ../data/preprocessed/split/fold{}.h5 \
    -o ../result/3mer_gcn_fold{} \
    -s ${seed} -d {d} ${no_fit} \
::: $(seq 0 4)
