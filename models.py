import collections
import numpy as np
from scipy.stats import hmean
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, sampler, Subset, RandomSampler, SequentialSampler

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

EvaluationStatistics = collections.namedtuple('EvaluationStatistics', ['tp', 'fp', 'tn', 'fn', 'predicted_count', 'actual_count'])
EvaluationResult = collections.namedtuple('EvaluationResult', ['precision', 'recall', 'accuracy', 'f1', 'kappa'])

def zero_statistics(n_classes):
    z = np.zeros_like(n_classes)
    return EvaluationStatistics(z, z, z, z, z, z)

def empty_results_history(n_classes):
    return EvaluationResult(precision = [[] for _ in range(n_classes)],
                            recall = [[] for _ in range(n_classes)],
                            accuracy = [],
                            f1 = [],
                            kappa = []
                           )

def record_results(history, results):
    c = len(history.precision)
    
    for i in range(c):
        history.precision[i].append(results.precision[i])
        history.recall[i].append(results.recall[i])
        
    history.accuracy.append(results.accuracy)
    history.f1.append(results.f1)
    history.kappa.append(results.kappa)

def compute_prediction_statistics(predicted : np.array, y : np.array, classes : list) -> EvaluationStatistics:
    return EvaluationStatistics(
        tp = np.array([np.logical_and(predicted == c, y == c).sum() for c in classes]),
        fp = np.array([np.logical_and(predicted == c, y != c).sum() for c in classes]),
        tn = np.array([np.logical_and(predicted != c, y != c).sum() for c in classes]),
        fn = np.array([np.logical_and(predicted != c, y == c).sum() for c in classes]),
        predicted_count = np.array([(predicted == c).sum() for c in classes]),
        actual_count = np.array([(y == c).sum() for c in classes]),
    )

def combine(s1 : EvaluationStatistics, s2 : EvaluationStatistics) -> EvaluationStatistics:
    return EvaluationStatistics(
        tp = s1.tp + s2.tp,
        fp = s1.fp + s2.fp,
        tn = s1.tn + s2.tn,
        fn = s1.fn + s2.fn,
        predicted_count = s1.predicted_count + s2.predicted_count,
        actual_count = s1.actual_count + s2.actual_count
    )

def evaluate(stats : EvaluationStatistics) -> EvaluationResult:
    EPS = 1e-6

    precision = stats.tp / (stats.tp + stats.fp)
    recall = stats.tp / (stats.tp + stats.fn)
    precision[np.where(np.isnan(precision))] = 1
    recall[np.where(np.isnan(recall))] = 1

    accuracy = (stats.tp.sum() + stats.tn.sum()) / (stats.tp.sum() + stats.tn.sum() + stats.fp.sum() + stats.fn.sum())

    f1 = hmean(np.concatenate([precision, recall]) + EPS)

    predicted_dist = stats.predicted_count / stats.predicted_count.sum()
    actual_dist = stats.actual_count / stats.actual_count.sum()
    expected_accuracy = np.sum(predicted_dist * actual_dist)
    kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
    
    return EvaluationResult(precision, recall, accuracy, f1, kappa)

@torch.no_grad()
def evaluate_model(model, dataset, classes, examples=None,
                   batch_size=16, dtype=torch.float32, device=DEFAULT_DEVICE):
    examples = min(examples, len(dataset)) or len(dataset)
    model.eval()
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler.SequentialSampler(range(examples or len(dataset))))
    
    stats = zero_statistics(len(classes))

    for x, y in loader:
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        scores = model(x).cpu().numpy()
        predictions = scores.argmax(axis=1)
        stats = combine(stats, compute_prediction_statistics(predictions, y.cpu().numpy(), classes))

    return evaluate(stats)

def format_metric(m):
    if np.isscalar(m):
        return '{:.3f}'.format(m)
    return '[' + ','.join(format_metric(mi) for mi in m) + ']'

def split_empty_nonempty(dataset):
    empty, nonempty = [], []
    for i in range(len(dataset)):
        (_, y) = dataset[i]

        if (y != 0).sum() > 0:
            nonempty.append(i)
        else:
            empty.append(i)
    return empty, nonempty

def train_model(model, train, val, learning_rate, batch_size=16, epochs=1,
          dtype=torch.float32, device=DEFAULT_DEVICE, verbose=False):

    model.cuda(device)
    
    _, nonempty_train = split_empty_nonempty(train)
    _, nonempty_val = split_empty_nonempty(val)

    if verbose:
        print(f'{len(nonempty_train)}/{len(train)} training examples are non-empty')
        print(f'{len(nonempty_val)}/{len(val)} validation examples are non-empty')

    nonempty_train_ds = Subset(train, nonempty_train)
    nonempty_val_ds = Subset(val, nonempty_val)

    train_loader = DataLoader(nonempty_train_ds,
                              batch_size=batch_size,
                              sampler=sampler.RandomSampler(nonempty_train_ds, True, len(train)))

    for l in model:
        if hasattr(l, '_should_init'):
            nn.init.kaiming_normal_(l.weight)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_history, train_history, val_history = [], empty_results_history(3), empty_results_history(3)
    
    it = 0

    for e in range(epochs):
        if verbose:
            print(f'Epoch {e+1}')
            
        n_iters = (len(train) + batch_size - 1) // batch_size

        for t, (x, y) in enumerate(train_loader):
            model.train()
            
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if it % 100 == 0 or (e + 1 == epochs and t + 1 == n_iters):
                train_result = evaluate_model(model, nonempty_train_ds, [0, 1, 2], 1000)
                record_results(train_history, train_result)

                val_result = evaluate_model(model, nonempty_val_ds, [0, 1, 2], 1000)
                record_results(val_history, val_result)
                
                if verbose:
                    print(f'Iteration {t}, loss={loss.item()}, ' +
                          f'prec={format_metric(train_result.precision)}/{format_metric(val_result.precision)}, ' +
                          f'recall={format_metric(train_result.recall)}/{format_metric(val_result.recall)}, ' +
                          f'acc={format_metric(train_result.accuracy)}/{format_metric(val_result.accuracy)}, ' +
                          f'f1={format_metric(train_result.f1)}/{format_metric(val_result.f1)}, ' +
                          f'kappa={format_metric(train_result.kappa)}/{format_metric(val_result.kappa)}, ')
                    
            it += 1

    return loss_history, train_history, val_history, nonempty_train
