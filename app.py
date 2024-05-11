from app.config import Configs
from app.logger import Logger

from app.inference import ModelInference

from app.cnn.unique_cnn import UniqueCNN, CNNUniqueConfig

from app.cnn.utils.preprocess_data import PreProcess
from app.cnn.utils.trainer import Trainer
import torch
def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ CIFAR-10 CNN PYTORCH CONFIGURATIONS ############")
    logger.debug(configs)

    if args.train:
        preprocess = PreProcess(**configs["preprocess"], logger=logger)
        train_loader , test_loader , train_dataset , test_dataset = preprocess()
        if not args.transfer_learning:

            config = CNNUniqueConfig(**configs["cnn-architecture"]["unique"])
            model = UniqueCNN(config)

        else:
            model = torch.load(configs["trainer"]["save_model_path"])
        
        post_process = Trainer(**configs["trainer"] , parameters = model.parameters(), logger=logger)

        post_process(model,train_loader,test_loader,train_dataset,test_dataset)

    if args.inference:
        inference = ModelInference(**configs["inference"], logger=logger)
        inference()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-tl", "--transfer-learning", action="store_true")
    

    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
