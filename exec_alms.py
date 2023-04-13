from sklearn.metrics import pairwise_distances
from ot import sliced_wasserstein_distance
from util import *


n_repeats = 10
n_cycles = 10
n_new_data = 3

idx_conds = [0, 1, 2, 3, 4, 5]
lb = [0.1, 850, 10, 0, 10, 3]
ub = [1, 1250, 350, 1, 50, 18]
path_dataset_init = 'datasets/dataset_init.xlsx'
path_dataset_un = 'datasets/dataset_un.xlsx'
path_dataset_test = 'datasets/dataset_test.xlsx'
path_dataset_origin = 'datasets/dataset.xlsx'
dataset_init, norm_mean, norm_std = load_dataset(path_dataset_init, path_dataset_origin, idx_conds, idx_target=10)
dataset_un, _, _ = load_dataset(path_dataset_un, path_dataset_origin, idx_conds, idx_target=10, norm_x=False)
_dataset_un_norm, _, _ = load_dataset(path_dataset_un, path_dataset_origin, idx_conds, idx_target=10)
dataset_test, _, _ = load_dataset(path_dataset_test, path_dataset_origin, idx_conds, idx_target=10)

repeat_mae_test = list()
repeat_r2_test = list()
repeat_wd_tr_new = list()
repeat_wd_test_new = list()


for n in range(0, n_repeats):
    torch.manual_seed(n)
    numpy.random.seed(n)
    random.seed(n)

    init_lr = 5e-4
    wd_prev = None
    init_lrs = list()
    list_mae_test = list()
    list_r2_test = list()
    list_indices = list()
    list_gen_data = list()
    wd_tr_new = list()
    wd_test_new = list()

    model, mae_test_init, r2_test_init = get_init_model('save/model_init_{}.pt'.format(n), dataset_test)
    am = ActiveMetaheuristic(deepcopy(dataset_init), deepcopy(dataset_test), model, lb, ub)
    dataset_un_norm = deepcopy(_dataset_un_norm)

    wd_test_new.append(sliced_wasserstein_distance(am.dataset_test, am.dataset_train))
    list_mae_test.append(mae_test_init)
    list_r2_test.append(r2_test_init)

    for i in range(0, n_cycles):
        new_x = am.search_new_data(norm_mean, norm_std, epoch=300).reshape(1, -1)
        pdists = pairwise_distances(normalize(new_x, norm_mean, norm_std), dataset_un_norm[:, :-1]).flatten()
        idx_new_data = numpy.argsort(pdists)[:n_new_data]
        new_data = dataset_un[idx_new_data]
        norm_data = dataset_un_norm[idx_new_data]
        dataset_un_norm = numpy.delete(dataset_un_norm, idx_new_data, axis=0)

        _dataset_train = deepcopy(am.dataset_train)
        _dataset_train_new = numpy.vstack([_dataset_train, norm_data])
        wd = sliced_wasserstein_distance(_dataset_train, _dataset_train_new)

        if wd > 0.1:
            am.append_train_data(norm_data)
            init_lr = 1e-5
        else:
            if i == 0:
                init_lr = 5e-4
            else:
                if wd_prev == 0:
                    init_lr = 1e-5
                else:
                    init_lr = numpy.maximum(init_lr * (wd / wd_prev), 1e-4)
                    init_lr = numpy.minimum(init_lr, 1e-3)

        list_gen_data.append(deepcopy(new_x))
        list_gen_data.append(deepcopy(new_data[:, :-1]))

        wd_prev = wd
        init_lrs.append(init_lr)
        wd_test = sliced_wasserstein_distance(am.dataset_test, am.dataset_train)
        wd_tr_new.append(wd)
        wd_test_new.append(wd_test)

        mae_test, r2_test, y_test, preds_test = am.train_model(init_lr=init_lr, epochs=1000)
        print('Repeat [{}/{}]\tIteration [{}/{}]\tMAE (C2): {:.4f}\tR2-score (C2): {:.4f}'
              .format(n + 1, n_repeats, i + 1, n_cycles, mae_test, r2_test))

        list_mae_test.append(mae_test)
        list_r2_test.append(r2_test)
        list_indices.append(idx_new_data)
        pred_results = pandas.DataFrame(numpy.hstack([y_test, preds_test]))
        pred_results.to_excel('save/alms/pred_results/preds_{}_{}.xlsx'.format(n, i), index=False, header=False)

    gen_data = pandas.DataFrame(numpy.vstack(list_gen_data))
    gen_data.to_excel('save/alms/gen_data/data_{}.xlsx'.format(n), index=False, header=False)
    torch.save(model.state_dict(), 'save/alms/model_{}.pt'.format(n))

    repeat_mae_test.append(list_mae_test)
    repeat_r2_test.append(list_r2_test)
    repeat_wd_tr_new.append(wd_tr_new)
    repeat_wd_test_new.append(wd_test_new)

print('================ C2 yield prediction ===================')
print('MAE (mean)', numpy.mean(repeat_mae_test, axis=0).tolist())
print('MAE (std.)', numpy.std(repeat_mae_test, axis=0).tolist())
print('R2 (mean)', numpy.mean(repeat_r2_test, axis=0).tolist())
print('R2 (std.)', numpy.std(repeat_r2_test, axis=0).tolist())
