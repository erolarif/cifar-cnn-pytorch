from dataclasses import dataclass
from typing import Any
from .cnn_architecture.cnn import CNN
from .cnn_architecture.cnn_2 import CNN2

from .cnn_architecture.cnn import CNNConfig
from .cnn_architecture.utils.preprocess_data import PreProcess
from .cnn_architecture.utils.post_process import PostProcess

@dataclass
class Train:
    cnn1_activation:bool
    cnn2_activation:bool
    configs: Any = None
    logger: Any = None


    def __post_init__(self)-> None:
        """
        Train the model according to selected CNN model
        """
        preprocess = PreProcess(**self.configs["preprocess"], logger=self.logger)
        train_loader , test_loader , train_dataset , test_dataset = preprocess()

        if self.cnn1_activation:
            config = CNNConfig(**self.configs["cnn-architecture"])
            model = CNN(config)
        if self.cnn2_activation:
            model = CNN2()

        post_process = PostProcess(**self.configs["postprocess"] , parameters = model.parameters(), logger=self.logger)

        post_process(model,train_loader,test_loader,train_dataset,test_dataset)