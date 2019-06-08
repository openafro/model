import torch
import torch.nn.functional as F  # useful stateless functions
from torch import nn, optim
from torch.utils.data import DataLoader, sampler, Subset, RandomSampler

def conv_relu(input_depth, output_depth,
              layers=2, kernel_size=5, n_filters=8):
    'Simple [Conv-ReLU]xN model'

    l = []

    for i in range(layers):
        l.append(nn.Conv2d(input_depth if i == 0 else n_filters,
                           output_depth if i == layers - 1 else n_filters,
                           kernel_size=kernel_size,
                           padding=(kernel_size // 2)))
        l[-1]._should_init = True

        if i < layers - 1:
            l.append(nn.GroupNorm(4, n_filters))
            l.append(nn.ReLU())

    return nn.Sequential(*l)

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataset, learning_rate, batch_size=16, epochs=1,
          dtype=torch.float32, device=DEFAULT_DEVICE, verbose=False):

    model.cuda(device)
    
    nonempty = []

    for i in range(len(dataset)):
        (_, y) = dataset[i]

        if (y != 0).sum() > 0:
            nonempty.append(i)

    if verbose:
        print(f'{len(nonempty)}/{len(dataset)} examples are non-empty')

    # nonempty = nonempty[:16]

    model.train()

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler.SubsetRandomSampler(range(len(dataset))))

    nonempty_ds = Subset(dataset, nonempty)
    nonempty_loader = DataLoader(nonempty_ds,
                                 batch_size=batch_size,
                                 sampler=sampler.RandomSampler(nonempty_ds, True, len(dataset)))

    for l in model:
        if hasattr(l, '_should_init'):
            nn.init.kaiming_normal_(l.weight)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_history = []

    for e in range(epochs):
        if verbose:
            print(f'Epoch {e+1}')
        for t, (x, y) in enumerate(nonempty_loader):
            x = x.to(device=device, dtype=dtype, copy=True, non_blocking=False)
            y = y.to(device=device, dtype=torch.long, copy=True, non_blocking=False)

            scores = model(x)

            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and t % 10 == 0:
                acc = (scores.argmax(dim=1) == y).to(dtype=torch.float32).mean()
                print(f'Iteration {t}, loss={loss.item()}, acc={acc}, fractions:',
                      (scores.argmax(dim=1) == 0).to(dtype=torch.float32).mean().item(),
                      (scores.argmax(dim=1) == 1).to(dtype=torch.float32).mean().item(),
                      (scores.argmax(dim=1) == 2).to(dtype=torch.float32).mean().item())

            loss_history.append(loss.item())

    return loss_history, nonempty
