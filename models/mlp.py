import torch.nn as nn
from base import BaseNN

#          (herencia)
class MLP(BaseNN): #Desacoplamiento 
    def __init__(self):
        super(MLP, self).__init__(name="mlp") #Llamada al constructor de la clase base
        self.network = nn.Sequential(
            nn.Flatten(), #Capa de aplanamiento
            nn.Linear(28 * 28, 128), # (caracteristicas de entrada, caracteristicas de salida)
            nn.ReLU(), 
            nn.Dropout(0.2), #regularización
            nn.Linear(128, 10), #logits
        )

    def forward(self, x):
        return self.network(x)

    