# -*- coding: utf-8 -*-
# code warrior: Barid
##########
from UNIVERSAL.data_and_corpus import offline_corpus, data_manager, dataset_preprocessing
import numpy as np
import tensorflow as tf
import configuration
from UNIVERSAL.basic_optimizer import learning_rate_op, optimizer_op
from UNIVERSAL.training_and_learning import callback_training
import lt
import os
import runai.ga.keras
import sys

cwd = os.getcwd()
#############
offline = [
    ["/home/vivalavida/massive_data/data/fair/wmt14_en_de_v3/train.en",],
    ["/home/vivalavida/massive_data/data/fair/wmt14_en_de_v3/train.de",],
]


def EOS_entailment(src, tgt):
    def _encode(lang1, lang2):
        def __plusEOS(x):
            x_eos = np.concatenate((x.numpy(), [configuration.parameters["EOS_ID"]]), 0)
            return x_eos

        x_eos = __plusEOS(lang1)
        y_eos = __plusEOS(lang2)
        return x_eos, y_eos

    x_eos, y_eos = tf.py_function(_encode, [src, tgt], [tf.int32, tf.int32,])
    x_eos.set_shape([None])
    y_eos.set_shape([None])
    return (x_eos, y_eos)


def preprocessed_dataset(shuffle=40):
    training_samples = offline_corpus.offline(offline)
    dataManager = data_manager.DatasetManager(cwd + "/../UNIVERSAL/vocabulary/DeEn_32000_v4/", training_samples)
    dataset = dataManager.get_raw_train_dataset()
    preprocessed_dataset = dataset_preprocessing.prepare_training_input(
        dataset,
        configuration.parameters["batch_size"],
        configuration.parameters["max_sequence_length"],
        min_boundary=8,
        filter_min=1,
        filter_max=configuration.parameters["max_sequence_length"],
        tf_encode=EOS_entailment,
        shuffle=shuffle,
    )
    return preprocessed_dataset, dataManager


def optimizer(tf_float16=True):
    tf.print("###################################", output_stream=sys.stderr)
    tf.print(
        "Gradients is applied affter: "
        + str(
            configuration.parameters["gradient_tower"]
            * configuration.parameters["batch_size"]
            # * configuration.parameters["max_sequence_length"]
        )
    )
    tf.print("###################################", output_stream=sys.stderr)
    opt = optimizer_op.MultistepAdamOptimizer(
        configuration.parameters["lr"],
        beta1=0.9,
        beta2=0.997,
        epsilon=1e-9,
        n=configuration.parameters["gradient_tower"],
        warmmup_steps=configuration.parameters["learning_warmup"],
        d_model=configuration.parameters["num_units"],
        init_step=50000,
    )
    return opt


def callbacks():
    return callback_training.get_callbacks(cwd, save_freq=10000)


def trainer():
    main_model = lt.LazyTransformer(configuration.parameters)
    return main_model
