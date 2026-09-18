import torch
import torch.nn as nn
from utils.device import get_device
from datasets.mnist import get_mnist_dataset
from models.mlp import MLP
from engine.trainer import fit, evaluate_one_epoch
from callbacks.early_stopping import EarlyStopping

def run_experiment(config):
    device = get_device()
    
    train_loader, val_loader, _ = get_mnist_dataset(
        data_dir='./data',
        batch_size=config['batch_size'],
        val_split=0.2
    )
    
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3)
    
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config.get('epochs', 10),
        early_stopping=early_stopping
    )
    
    val_loss = evaluate_one_epoch(model, val_loader, criterion, device)
    return val_loss