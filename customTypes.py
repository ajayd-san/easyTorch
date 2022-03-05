from typing import Union, Tuple, List, Callable, Dict

import numpy as np
import torch
from torch.utils import data

TensorOrArray = Union[torch.Tensor, np.ndarray]
ImagePlusTarget = Tuple[TensorOrArray, TensorOrArray]
MetricList = List[Tuple[Callable, Dict]]
Dataloader = data.DataLoader
