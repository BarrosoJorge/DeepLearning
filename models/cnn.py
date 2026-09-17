import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), #color, no.kernel, tamanho do kernel, saltos, agregar capa de pixeles
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,), 
            #La salida son mapas de carateristicas
            nn.Dropout(0,1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #entrada, salida, tamanho do kernel, saltos, agregar capa de pixeles
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.2), #En entrenamiento, se apaga ese % de la muestra
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x #logits - Datos crudos de salida en capa