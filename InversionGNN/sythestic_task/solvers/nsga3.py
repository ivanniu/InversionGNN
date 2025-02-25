from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def f1(x):
    n = x.shape[1]
    dx = np.linalg.norm(x - 1. / np.sqrt(n), axis=1, keepdims=True)
    return 1 - np.exp(-dx**2)


def f2(x):
    n = x.shape[1]
    dx = np.linalg.norm(x + 1. / np.sqrt(n), axis=1, keepdims=True)
    return 1 - np.exp(-dx**2)


class MyProblem(Problem):
    def __init__(self, r = None, multi_obj_fg = None):
        super().__init__(n_var=20,    
                         n_obj=2,    
                         n_constr=0, 
                         xl= -  np.ones(20), 
                         xu=   np.ones(20),
                         )    
        self.r = r
        self.multi_obj_fg = multi_obj_fg
        self.eval_count = 0
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x / 0.001) * 0.001 
        fv1 = f1(x)
        fv2 = f2(x)
        fv = np.hstack((fv1,fv2))
        out["F"] = fv
        self.eval_count += 1

def nsga3_search(multi_obj_fg, r, max_iters=100):
    
    problem = MyProblem(r,multi_obj_fg)

    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=4)
    #print(ref_dirs)

    # ref_dirs = np.array(r)
    # ref_dirs = ref_dirs / np.sum(ref_dirs, axis=1)[:, np.newaxis]

    #ref_dirs = np.array([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8], [0.0, 1.0]])

    #ref_dirs = np.array([[0.86, 0.14], [0.66, 0.34], [0.5, 0.5], [0.34, 0.66], [0.34, 0.66], [0.14, 0.86]])
    algorithm = NSGA3(
        pop_size=40,
        n_offsprings=10,
        ref_dirs=ref_dirs,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    full_draw = []

  
    res = minimize(problem,
            algorithm,
            ("n_gen", max_iters),
            verbose=True,
            seed=1,
            save_history=True)
    full_draw.append(res.F)

  
    

    return full_draw

