import torch
import math
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()

####################dataset######################################
def Brelu(x):
    out = min(max(x,0),1)
    return  out

print(Brelu(0.01))