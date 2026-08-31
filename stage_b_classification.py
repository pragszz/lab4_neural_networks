from torch.utils.data import DataLoader
import torch.nn as nn
batch_size = 32


def build_mlp_img():
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
    )
    return model

def load_fn(data):
    dataset = DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True  
    )
    
    return dataset