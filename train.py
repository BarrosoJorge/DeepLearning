import torch
import torch.nn as nn
from utils.device import get_device
from datasets.mnist import get_mnist_dataset
from models.mlp import MLP
from engine.trainer import fit, evaluate_one_epoch
from utils.plotting import plot_history
from callbacks.early_stopping import EarlyStopping


def main() -> None:
    #hiperparámetros
    num_epochs = 100 #callback --> early stopping (otra forma de regularización)
    learning_rate = 0.1

    '''
    schedulers: 

    * StepLR: reduce el learning rate en pasos fijos (cada n épocas)
    * ExponentialLR: reduce el learning rate de forma exponencial
    * ReduceLROnPlateau: reduce el learning rate cuando la métrica de validación deja de mejorar
    
    '''

    batch_size = 64 #numero de muestras por batch

    patience = 6 #número de épocas sin mejora antes de detener el entrenamiento
    min_delta = 1e-3 #mejora mínima para considerar que hay mejora

    device = get_device() #obtener el dispositivo a usar
    print(f'Entrenando con: {device}')

    train_loader, val_loader, test_loader = get_mnist_dataset(
        data_dir='./data', 
        batch_size=batch_size,
        val_split=0.2
        ) #obtener los dataloaders

    ##modelo
    model = MLP()
    model = model.to(device) #mover el modelo al dispositivo

    #entrenamiento
    criterion = nn.CrossEntropyLoss() #función de pérdida
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        #momentum = 0.9,
        ) #optimizador

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, # que optimizador se va a usar
        mode='min', # minimizar la métrica de validación
        factor=0.1, # factor de reducción del learning rate
        patience=2, # número de épocas sin mejora antes de reducir el learning rate
    ) #scheduler para reducir el learning rate cuando la métrica de validación deja de mejorar

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) #callback de early stopping

    history = fit(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs, 
        early_stopping
        #scheduler
    )

    torch.save(model.state_dict(), 'artifacts/best_model.pth') #guardar el mejor modelo

    test_loss = evaluate_one_epoch(
        model, 
        test_loader, 
        criterion, 
        device
    )


    print(f'Test Loss: {test_loss:.4f}')
    plot_history(history) #graficar las pérdidas de entrenamiento y validación

if __name__ == '__main__':
    main()