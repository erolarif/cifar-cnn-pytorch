from app.config import Configs
from app.logger import Logger

from app.train import Train
from app.inference import ModelInference
def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ CIFAR-10 CNN PYTORCH CONFIGURATIONS ############")
    logger.debug(configs)

    if args.train:
        Train(**configs["train"], configs=configs, logger=logger)

    if args.inference:
        inference = ModelInference(**configs["inference"], logger=logger)
        inference()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-i", "--inference", action="store_true")

    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
