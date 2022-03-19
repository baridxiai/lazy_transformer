# -*- coding: utf-8 -*-
# code warrior: Barid

import contextlib
import tensorflow as tf
import atexit
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  # fp16 training
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(0)


# import argparse
@contextlib.contextmanager
def config_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


options = {
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": True,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": True,
    "disable_meta_optimizer": True,
    "min_graph_nodes": True,
}
config_options(options)


def main():
    import initialization

    tf.print("Let's go Celtics", output_stream=sys.stderr)
    tf.print("###################################", output_stream=sys.stderr)
    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore
    with strategy.scope():
        data_opt = tf.data.Options()
        data_opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # data_opt.autotune.enabled = True
        # train_dataset,_ = initialization.preprocessed_dataset()
        model = initialization.trainer()
        optimizer = initialization.optimizer()
        callbacks = initialization.callbacks()
        # uncomment for training###
        try:
            model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
        except Exception:
            pass
        ##################
        model.compile(optimizer=optimizer)
        tf.print("Checking model!", output_stream=sys.stderr)
        tf.print("###################################", output_stream=sys.stderr)
        # model.test_on_batch(np.array([[2]]))
        # model.summary(print_fn=tf.print)
        train_dataset, _ = initialization.preprocessed_dataset(shuffle=5000000)
        train_dataset = train_dataset.with_options(data_opt)
        # import pdb; pdb.set_trace()
        # model.build(None)
        # model.summary()
        model.fit(
            train_dataset, epochs=initialization.configuration.parameters["epoch"], verbose=1, callbacks=callbacks
        )


if __name__ == "__main__":
    main()
