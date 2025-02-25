import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def f1(x):
    n = x.shape[1]
    dx = torch.linalg.norm(x - 1. / np.sqrt(n), axis=1, keepdims=True)
    return 1 - torch.exp(-dx**2)


def f2(x):
    n = x.shape[1]
    dx = torch.linalg.norm(x + 1. / np.sqrt(n), axis=1, keepdims=True)
    return 1 - torch.exp(-dx**2)

def your_function(x):
    return -torch.hstack((f1(x),f2(x)))


def mobo_search():
    # Problem settings
    dim = 20
    bounds = torch.stack([- torch.ones(dim), torch.ones(dim)])

    # Generate initial data

    train_X = draw_sobol_samples(bounds=bounds, n=1, q=1000, seed=0).squeeze(0)

    train_Y = your_function(train_X)
    # Define model
    models = []
    for i in range(train_Y.shape[-1]):
        models.append(SingleTaskGP(train_X, train_Y[..., i : i + 1]))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)


    # Define acquisition function
    ref_point = torch.tensor([-1, -1], dtype=torch.double)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y),
        sampler=None,
    )

    # Optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=10,
        num_restarts=5,
        raw_samples=20,
    )

    for t in range(100):
        new_Y = your_function(candidates)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])

        models = []
        for i in range(train_Y.shape[-1]):
            models.append(SingleTaskGP(train_X, train_Y[..., i : i + 1]))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y),
        sampler=None,)



        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=10,
            num_restarts=5,
            raw_samples=20,
        )

        print(f"Iteration {t+1}, candidates: {candidates}")
    
    nds = NonDominatedSorting().do(-train_Y.numpy(), only_non_dominated_front=True)
    nds_train_Y = train_Y[nds,:]
    return train_Y, nds_train_Y