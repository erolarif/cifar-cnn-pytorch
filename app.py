from app.config import Configs
from app.logger import Logger
from app.cnn_architecture.cnn import CNN
from app.cnn_architecture.cnn_2 import CNN2

from app.cnn_architecture.cnn import CNNConfig
from app.cnn_architecture.utils.preprocess_data import PreProcess
from app.cnn_architecture.utils.post_process import PostProcess


def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ CIFAR-10 CNN PYTORCH CONFIGURATIONS ############")
    logger.debug(configs)

    preprocess = PreProcess(**configs["preprocess"], logger=logger)
    train_loader , test_loader , train_dataset , test_dataset = preprocess()
    config = CNNConfig(**configs["cnn-architecture"])
    # model = CNN(config)
    model = CNN2()

    post_process = PostProcess(**configs["postprocess"] , parameters = model.parameters(), logger=logger)

    post_process(model,train_loader,test_loader,train_dataset,test_dataset)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
