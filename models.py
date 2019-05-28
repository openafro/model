import torch
import torch.nn.functional as F  # useful stateless functions
from torch import nn, optim
from torch.utils.data import DataLoader, sampler

def conv_relu(input_depth, output_depth,
              layers=2, kernel_size=5, n_filters=8):
    'Simple [Conv-ReLU]xN model'

    l = []

    for i in range(layers):
        l.append(nn.Conv2d(input_depth if i == 0 else n_filters,
                           output_depth if i == layers - 1 else n_filters,
                           kernel_size=kernel_size,
                           padding=(kernel_size // 2)))
        if i < layers - 1:
            l.append(nn.ReLU())

    return nn.Sequential(*l)

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataset, learning_rate, batch_size=16, epochs=1,
          dtype=torch.float32, device=DEFAULT_DEVICE):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler.SubsetRandomSampler(range(len(dataset))))

    model.train()

    for l in model:
        if hasattr(l, 'weight'):
            nn.init.normal_(l.weight, std=1e-1)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_history = []

    for e in range(epochs):
        print(f'Epoch {e+1}')
        for t, (x, y) in enumerate(loader):

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype).argmax(dim=1)  # move to device, e.g. GPU

            scores = model(x)

            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 10 == 0:
                acc = (scores.argmax(dim=1) == y).to(dtype=torch.float32).mean()
                print(f'Iteration {t}, loss={loss.item()}, acc={acc}, fractions:',
                      (scores.argmax(dim=1) == 0).to(dtype=torch.float32).mean().item(),
                      (scores.argmax(dim=1) == 1).to(dtype=torch.float32).mean().item(),
                      (scores.argmax(dim=1) == 2).to(dtype=torch.float32).mean().item())

            loss_history.append(loss.item())

    return loss_history
