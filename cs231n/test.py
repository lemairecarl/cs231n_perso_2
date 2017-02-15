import numpy as np
from cs231n.classifiers.convnet import *

model = FlexNet(num_filters=(32, 64, 128), hidden_dim=(100, 50))
