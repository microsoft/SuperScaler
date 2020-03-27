import os
import logging
from argparse import ArgumentParser
from adapter.tf_adapter import TFAdapter
from profiler.profiler import Profiler
from simulator import Simulator

def simulate_single_model(model_file_path):
    logger = logging.getLogger('simulator_app')
    logger.info("Simulate model for %s ..." %model_file_path)

    if not model_file_path.endswith(".pbtxt"):
        logger.error('Input model file %s can not be parsed!' % model_file_path)
        return
    test_tf_adapter =  TFAdapter()
    # Parse the tensorflow graph
    test_graph = test_tf_adapter.parse_protobuf_graph(test_tf_adapter.load_pbtxt_file(model_file_path))
    # rofiling for every node
    for node in test_graph:
        node.execution_time = Profiler().get_node_execution_time(node)
    # Run the simulator
    sim = Simulator(test_graph)
    timeuse, execution_list = sim.run()
    logger.info("Simulator time: %.3f in total." %timeuse)


def main():
    parser = ArgumentParser()
    parser.add_argument("-g","--graph",help="The path to the model (either a directory or a file)",default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "examples")))
    args = parser.parse_args()

    logger = logging.getLogger('simulator_app')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('./simulator_app.log',mode='w')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Simulator app logging starts ...')

    if os.path.isfile(args.graph):
        simulate_single_model(args.graph)
    elif os.path.isdir(args.graph):
        for subdir, dirs, files in os.walk(args.graph):
            for file in files:
                model_path = os.path.join(subdir, file)
                simulate_single_model(os.path.join(subdir, file))
    else:
        logger.error("Input model path does not exist!")

    logger.removeHandler(fh)
    logger.removeHandler(ch)
    fh.close()
    ch.close()
        

if __name__ == "__main__":
    main()

