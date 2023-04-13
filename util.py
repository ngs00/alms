import numpy
import pandas
import torch
import random
import torch.nn as nn
import mealpy.utils.problem
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from mealpy.swarm_based.ABC import OriginalABC


class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.emb_layer = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.std_layer = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        embs = self.emb_layer(x)
        means = self.mean_layer(embs)
        stds = self.std_layer(embs)

        return torch.distributions.Normal(means, stds)

    def fit(self, data_loader, optimizer):
        sum_losses = 0

        self.train()
        for x, y in data_loader:
            x = x.cuda()
            y = y.cuda()

            preds = self(x)
            loss = torch.mean(-preds.log_prob(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            preds = self(torch.tensor(x, dtype=torch.float).cuda())

            return preds.mean.cpu().numpy(), preds.stddev.cpu().numpy()

    def pred_uncert(self, x):
        self.eval()

        with torch.no_grad():
            preds = self(torch.tensor(x, dtype=torch.float))

            return preds.stddev.numpy()


class Problem(mealpy.utils.problem.Problem):
    def __init__(self, model, lb, ub, norm_mean, norm_std, **kwargs):
        self.model = model
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        super().__init__(lb, ub, minmax='max', **kwargs)

    def fit_func(self, sol):
        _sol = numpy.clip(sol, self.lb, self.ub)
        _sol = normalize(_sol, self.norm_mean, self.norm_std)
        return self.model.pred_uncert(_sol)


class ActiveMetaheuristic:
    def __init__(self, dataset_init, dataset_test, model, lb, ub):
        self.dataset_init = dataset_init
        self.dataset_train = deepcopy(self.dataset_init)
        self.dataset_test = dataset_test
        self.model = model
        self.lb = lb
        self.ub = ub

    def train_model(self, batch_size=32, init_lr=5e-4, r2_coeff=5e-6, epochs=1000):
        dataset_train_x = torch.tensor(self.dataset_train[:, :-1], dtype=torch.float)
        dataset_train_y = torch.tensor(self.dataset_train[:, -1], dtype=torch.float).view(-1, 1)
        dataset_test_y = self.dataset_test[:, -1].reshape(-1, 1)
        loader_train = DataLoader(TensorDataset(dataset_train_x, dataset_train_y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, weight_decay=r2_coeff)

        self.model.cuda()
        for epoch in tqdm(range(0, epochs)):
            self.model.fit(loader_train, optimizer)

        preds_test, _ = self.model.predict(self.dataset_test[:, :-1])
        mae_test = mean_absolute_error(dataset_test_y, preds_test)
        r2_test = r2_score(dataset_test_y, preds_test)

        return mae_test, r2_test, dataset_test_y, preds_test

    def search_new_data(self, norm_mean, norm_std, epoch=1000):
        problem = Problem(self.model.cpu(), self.lb, self.ub, norm_mean, norm_std)
        opt = OriginalABC(epoch=epoch)
        new_data, _ = opt.solve(problem=problem)

        return new_data

    def append_train_data(self, new_data):
        self.dataset_train = numpy.vstack([self.dataset_train, new_data])


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(z, mean, std):
    return std * z + mean


def load_dataset(path_dataset, path_dataset_origin, idx_feats, idx_target, norm_x=True):
    dataset = numpy.array(pandas.read_excel(path_dataset))
    dataset_origin = numpy.array(pandas.read_excel(path_dataset_origin))
    dataset_x = dataset[:, idx_feats]
    dataset_y = dataset[:, idx_target].reshape(-1, 1)
    dataset_origin_x = dataset_origin[:, idx_feats]
    norm_mean = None
    norm_std = None

    if norm_x:
        norm_mean = numpy.mean(dataset_origin_x, axis=0)
        norm_std = numpy.std(dataset_origin_x, axis=0)
        dataset_x = normalize(dataset_x, norm_mean, norm_std)

    return numpy.hstack([dataset_x, dataset_y]), norm_mean, norm_std


def save_init_model(dataset_init, dataset_test, n_repeat):
    lb = [0.1, 850, 10, 0, 10, 3]
    ub = [1, 1250, 350, 1, 50, 18]

    torch.manual_seed(n_repeat)
    numpy.random.seed(n_repeat)
    random.seed(n_repeat)

    model = BNN().cuda()
    am = ActiveMetaheuristic(dataset_init, dataset_test, model, lb, ub)
    mae_test_init, r2_test_init, y_test, preds_test = am.train_model(init_lr=1e-2, epochs=100)
    pred_results = pandas.DataFrame(numpy.hstack([y_test, preds_test]))
    pred_results.to_excel('save/preds_init_{}.xlsx'.format(n_repeat), index=False, header=False)
    torch.save(model.state_dict(), 'save/model_init_{}.pt'.format(n_repeat))
    print('Initial iteration\tMAE (C2): {:.4f}\tR2-score (C2): {:.4f}'.format(mae_test_init, r2_test_init))


def get_init_model(path_init_model, dataset_test):
    model = BNN().cuda()
    model.load_state_dict(torch.load(path_init_model))

    targets_test = dataset_test[:, -1].reshape(-1, 1)
    preds_test, _ = model.predict(dataset_test[:, :-1])
    mae_test = mean_absolute_error(targets_test, preds_test)
    r2_test = r2_score(targets_test, preds_test)

    return deepcopy(model), mae_test, r2_test
