from dataclasses import dataclass
from typing import Any
from .cnn_architecture.unique_cnn import UniqueCNN, CNNUniqueConfig
from .cnn_architecture.optimized_cnn import OptimizedCNN, CNNOptimizedConfig

from .cnn_architecture.utils.preprocess_data import PreProcess
from .cnn_architecture.utils.post_process import PostProcess

@dataclass
class Train:
    unique_cnn_activation:bool
    optimized_cnn_activation:bool
    configs: Any = None
    logger: Any = None


    def __post_init__(self)-> None:
        """
        Train the model according to selected CNN model
        """
        preprocess = PreProcess(**self.configs["preprocess"], logger=self.logger)
        train_loader , test_loader , train_dataset , test_dataset = preprocess()

        if self.unique_cnn_activation:
            config = CNNUniqueConfig(**self.configs["cnn-architecture"]["unique"])
            model = UniqueCNN(config)
        if self.optimized_cnn_activation:
            config = CNNOptimizedConfig(**self.configs["cnn-architecture"]["optimized"])
            model = OptimizedCNN(config)

        post_process = PostProcess(**self.configs["postprocess"] , parameters = model.parameters(), logger=self.logger)

        post_process(model,train_loader,test_loader,train_dataset,test_dataset)