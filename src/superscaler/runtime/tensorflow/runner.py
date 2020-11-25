# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import argparse
import superscaler.runtime.tensorflow.runtime as rt
import tensorflow as tf


# Training settings
parser = argparse.ArgumentParser(description='Runner')
parser.add_argument('--model_dir_prefix', type=str, required=True,
                    help='Resource directory patth ')
parser.add_argument('--steps', type=int, default=10,
                    help='number of steps to train (default: 10)')
parser.add_argument('--interval', type=int, default=5,
                    help='number of interval to print info (default: 5)')
parser.add_argument('--print_info', type=bool, default=False,
                    help='Enable printing infomation (default: False)')
parser.add_argument('--print_fetches_targets', type=bool, default=False,
                    help='Enable printing fetches_targets (default: False)')


def main():
    args = parser.parse_args()
    # prepare resource
    pbtxt_path = os.path.join(args.model_dir_prefix, 'graph.pbtxt')
    desc_path = os.path.join(args.model_dir_prefix, 'model_desc.json')
    plan_path = os.path.join(args.model_dir_prefix, 'plan.json')
    lib_path = os.path.abspath(os.path.join(os.environ["SUPERSCLAR_PATH"],
                                            "lib/libtfadaptor.so"))

    # Superscaler: initialize runtime library.
    sc = rt.TFRuntime(pbtxt_path, desc_path, plan_path, lib_path)

    # Superscaler: pin tensorflow configure
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    # The session initialization is created by given model protobuf.
    # Init operators and fetches operators are runned during training.
    with tf.compat.v1.Session(graph=sc.graph, config=config) as sess:
        sess.run(sc.inits)
        start = time.time()
        for step in range(1, args.steps + 1):
            fetches_targets = sess.run(sc.fetches + sc.targets)

            # Logging infos
            if step % args.interval == 0 and args.print_info:
                end = time.time()
                # Print Runtime information
                print("Runtime on host %s device %s "
                      "between step %d-%d : %f sec/step.\n" %
                      (sc.host_id(),
                       sc.device_id(),
                       step - args.interval,
                       step,
                       (end-start)/args.interval))

                # Print fetches_targets information
                if args.print_fetches_targets:
                    for name, info in zip(sc.fetches + sc.targets,
                                          fetches_targets):
                        print(name, info)
                start = time.time()


if __name__ == '__main__':
    main()
