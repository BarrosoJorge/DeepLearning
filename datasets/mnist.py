from os.path import split
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms

def get_mnist_dataset(data_dir, batch_size=64, val_split = 0.2): #Obtener los datos de entrenamiento
	#Data augmentation - Aumentar los datos de entrenamiento
	train_transforms = transforms.Compose([
		transforms.RandomRotation(degrees=10), #Rotar la imagen
		transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), #Trasladar la imagen
		transforms.ToTensor(), #Convertir a tensor
		#MNIST media = 0.1307, std = 0.3081
		transforms.Normalize((0.1307,), (0.3081,)) #Normalizar la imagen
	])
	eval_transforms = transforms.Compose([
			transforms.ToTensor(), #Convertir a tensor
			#MNIST media = 0.1307, std = 0.3081
			transforms.Normalize((0.1307,), (0.3081,)) #Normalizar la imagen
		])

	#transform = transforms.ToTensor()
	full_train_dataset = datasets.MNIST(
		data_dir,
		train=True,
		download=True,
		transform=train_transforms,
	)

	test_dataset = datasets.MNIST( #Obtener los datos de prueba
		data_dir,
		train=False,
		download=True,
		transform=eval_transforms,
	)

	val_size = int(len(full_train_dataset) * val_split)
	train_size = len(full_train_dataset) - val_size
	train_dataset, val_dataset = random_split(
		full_train_dataset,
		[train_size, val_size],
		generator=torch.Generator().manual_seed(42)
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size = batch_size,
		shuffle=False
	)
	val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader