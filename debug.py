# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import sys
import os
import numpy as np
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.compat.v1.disable_eager_execution()
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
with tf.device("/CPU:0"):
    import initialization

    optimizer = initialization.optimizer()
    model = initialization.trainer()
    model.compile(optimizer=optimizer)
    train_data, data_manager = initialization.preprocessed_dataset(1)
    try:
        model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
    except Exception:
        pass
    for index, inputs in enumerate(train_data.take(5)):
        from UNIVERSAL.basic_metric import bleu_metric
        from UNIVERSAL.evaluation.get_test_data import get_input_and_hyp
        from UNIVERSAL.evaluation.on_the_fly_bleu import evl_pack, zero_shot_inferring, report_bleu

        m_real_x = data_manager.encode(
            "The system is fitted with coloured LEDs , which are bright enough that drivers can easily see the lights , even when the sun is low in the sky . [EOS]"
        )
        reference = data_manager.encode(
            "Die Anlage ist mit farbigen LEDs ausgestattet , die so kräftig leuchten , dass die Lichter von den Autofahrern beispielsweise auch bei tiefstehender Sonne gut zu erkennen sind . [EOS]"
        )

        de_real_y = tf.pad([reference], [[0, 0], [1, 0]], constant_values=1)[:, :-1]
        tgt = model.call(([m_real_x], de_real_y), training=True, sos=1, eos=2, vis=True)
        ######################################################################################################
        m_real_x, reference = get_input_and_hyp(
            "/home/vivalavida/massive_data/data/fair/test.en",
            "/home/vivalavida/massive_data/data/fair/test.de",
            span=1,
            add_eos=True,
        )
        m_real_x = [data_manager.encode(i) for i in m_real_x]
        ref = [data_manager.encode(i) for i in reference]
        fn_model = partial(
            model,
            enc=initialization.configuration.parameters["num_encoder_steps"],
            dec=initialization.configuration.parameters["num_decoder_steps"],
        )
        tf.print("##########################################################################")
        tf.print(report_bleu(model, m_real_x, ref, eos_id=2, data_manager=data_manager))
        tf.print(
            zero_shot_inferring(
                model,
                m_real_x,
                ref,
                enc_range=12,
                dec_range=12,
                enc_offset=34,
                dec_offset=0,
                enc_static=50,
                dec_static=20,
                eos_id=2,
                data_manager=data_manager,
            )
        )
        break
        #############################################################################
    print("#####################")
