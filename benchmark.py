import argparse

import torch
from torch import nn

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

class MLP(nn.Module):
    def __init__(self, input_size, n_classes) -> None:
        super().__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size[0] * input_size[1], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

        # Activations
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.2),
            self.fc2,
            nn.ReLU(),
            nn.Dropout(0.2),
            self.fc3
        )

    def forward(self, x, sparse=False):
        if not sparse:
            return self.model(x.flatten())
        else:
            return self.model(x)

def main(opt):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Build a simple MLP model
    model = MLP(input_size=(28, 28), n_classes=10)

    # Generate a random sparse matrix
    sparsity = 0.95
    m = torch.rand((28, 28))
    mask = m > 0.95
    dense = m * mask
    sparse = dense.to_sparse_csr()

    # Run the model
    model(sparse, sparse=True)

if __name__ == '__main__':
    opt = parser_opt()
    main(opt)