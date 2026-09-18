import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNN
from torchvision.models import vgg11, VGG11_Weights

class VGG11(BaseNN):
    def __init__(self):
        super(VGG11, self).__init__(name="vgg11")
        self.network = vgg11(weights=VGG11_Weights.DEFAULT)

        
