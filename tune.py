import os
import json
import random
from tuning.searchers import random_search
from tuning.objective import run_experiment

def main():
    # Espacio de búsqueda
    param_distributions = {
        'learning_rate': lambda: 10 ** random.uniform(-4, -2),
        'batch_size': [32, 64, 128],
        'weight_decay': [0.0, 1e-4, 1e-3],
        'epochs': [8]
    }

    n_iter = 5
    best_loss = float('inf')
    best_config = None

    print(f"Iniciando Búsqueda Aleatoria ({n_iter} iteraciones)...\n")

    for i, config in enumerate(random_search(param_distributions, n_iter=n_iter), start=1):
        print(f"--- Prueba {i}/{n_iter}: {config} ---")
        val_loss = run_experiment(config)
        print(f"Validation Loss: {val_loss:.4f}\n")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config

    print("=" * 40)
    print(f"Mejor combinación: {best_config}")
    print(f"Mejor Validation Loss: {best_loss:.4f}")
    print("=" * 40)

    # Guardar los mejores hiperparámetros
    os.makedirs('artifacts', exist_ok=True)
    save_path = os.path.join('artifacts', 'best_hyperparameters.json')
    
    with open(save_path, 'w') as f:
        json.dump(best_config, f, indent=4)
        
    print(f"Hiperparámetros guardados exitosamente en: {save_path}")

if __name__ == '__main__':
    main()