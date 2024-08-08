## RoBERTa approach
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaConfig, RobertaModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Función de enmascarado para series temporales
def mask_data(data, mask_prob=0.15):
    mask = torch.full(data.shape, mask_prob, device=data.device)
    mask = torch.bernoulli(mask).bool()
    masked_data = data.clone()
    masked_data[mask] = 0  # Asignar cero a los valores enmascarados
    return masked_data, mask


data = np.load('../tscds.npy')  # Cargar los datos desde el archivo .npy
data = torch.tensor(data, dtype=torch.float32).to(device)  # Convertir los datos a un tensor de PyTorch

batch_size = 128  # Definir el tamaño del batch
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


config = RobertaConfig(
    hidden_size=512,  # Ajustar el tamaño de embeddings a 512
    num_hidden_layers=12,  # Mantener el mismo número de capas
    num_attention_heads=8,  # Ajustar el número de cabezas de atención proporcionalmente
    intermediate_size=2048  # Mantener el tamaño de la capa intermedia
)
model = RobertaModel(config).to(device)

print(model)

print(model.num_parameters())

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Por ejemplo, MSE para series temporales
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Paso 4: Entrenamiento (esquemático)
for epoch in range(100):  # Número de épocas
    for batch in dataloader:
        masked_data, mask = mask_data(batch[0])  # Enmascarar datos
        optimizer.zero_grad()
        outputs = model(inputs_embeds=masked_data).last_hidden_state # se utiliza input_embeds para pasar directametne los embeddings
        loss = criterion(outputs[mask], batch[0][mask])  # Comparar solo los valores enmascarados
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Entrenamiento completado")
torch.save(model, 'roberta_pretrained_model.pth')