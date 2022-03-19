# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.utils import checkMachine_util

GPU = 2
parameters = {
    "name": "LazyTransformer",
    ### small ##########
    # "norm_dropout":0.1,
    # "num_units": 512,
    # "num_heads": 8,
    # "embedding_size": 512,
    ### base ##########
    "norm_dropout": 0.3,
    "num_units": 1024,
    "num_heads": 16,
    "embedding_size": 1024,
    #####################
    "num_encoder_steps": 40,
    "num_decoder_steps": 6,
    "dropout": 0.1,
    ####### control update frequency  ##############
    ##################  32K is common for  base, 64k is common for big
    "batch_size": 1000 * GPU,
    "gradient_tower": 7,
    "max_sequence_length": 256,
    "epoch": 30,
    ####################################
    "PAD_ID": 0,
    "SOS_ID": 1,
    "EOS_ID": 2,
    "UNK_ID": 3,
    "MASK_ID": 4,
    "lr": 2,
    "learning_warmup": 8000,
    "vocabulary_size": 32010,
    "epsilon": 1e-9,
    "clip_norm": 0.0,
    "preNorm": True,
    "label_smoothing": 0.1,
    "step_encoding": False,
    "position_encoding": True,
    "scale_we": True,
    "affine_we": False,
    "ffn_activation": "relu",
    "beam_size": 4,
    "alpha": 0.6,
}
