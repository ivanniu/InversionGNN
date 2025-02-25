import numpy as np

from problems.toy_biobjective import circle_points, concave_fun_eval, create_pf
from solvers import pci_search, linscalar , mobo_search, nsga3_search


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import latexipy as lp
from pymoo.factory import get_performance_indicator

if __name__ == '__main__':
    is_tarck = False
    lp.latexify()
    if not is_tarck:
        K = 5       # Number of trajectories
    else:
        K = 200
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K,np.pi / 100,np.pi*49/100)  # preference

    pmtl_K = 5
    pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)
    methods = ['NSGA3']
    #methods = ['MOBO']
    #methods = ['LinScalar']
    #methods = ['Inversion']
    colors = ['dodgerblue','orange','green','firebrick','slateblue']
    nsga_colors = ['lightsteelblue','cornflowerblue','royalblue','blue']
    plt.rcParams["figure.figsize"] = (2, 1.55)  
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7 
    ss = 0.1
    pf = create_pf()
    with lp.figure('Inversion'):
        for method in methods:
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)
            ax.plot(pf[:, 0], pf[:, 1], lw=1.5, c='k')
            last_ls = []
            tarck_x = []
            tarck_y = []
            for k, r in enumerate(rs):
                if not is_tarck:
                    r_inv = 1. / r
                    ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
                    ep_ray_line = np.stack([np.zeros(m), ep_ray])
                    label = r'$r^{-1}$ ray' if k == 0 else ''
                    ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                            lw=1, ls='--', dashes=(10, 5))
                    ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                            .05 * ep_ray[0], .05 * ep_ray[1],
                        color='k', lw=1, head_width=.02)
                    if k == 0: 
                        ax.text(.5 * ep_ray[0]+0.1, .5 * ep_ray[1],r'$\lambda^{-1}$ Ray',fontsize=8,c='k')
                        ax.text(0.7, 0,'Pareto front',fontsize=8,c='k')
                x0 = np.zeros(n)
                x0[range(0, n, 2)] = 0.3
                x0[range(1, n, 2)] = -.3
                x0 += 0.1 * np.random.randn(n)
                x0 = np.random.uniform(-0.6, 0.6, n) if method in ["MOOMTL", "LinScalar"] else x0
                x0 = np.round(x0 / 0.001) * 0.001
                if method == 'Inversion':
                    _, res = pci_search(concave_fun_eval, r=r, x=x0,
                                        step_size=ss, max_iters=100)

                if method == 'LinScalar':
                    _, res = linscalar(concave_fun_eval, r=r, x=x0,
                                    step_size=ss, max_iters=150)

                if method in ['Inversion', 'LinScalar']:
                    if not is_tarck:
                        ax.scatter(res['ls'][:, 0], res['ls'][:, 1], s=6, c=colors[k], alpha=1,label=r'$\alpha =$ {:.2f}'.format(r[0]/r[1]))
                    else:
                        tarck_x.append(res['ls'][-1, 0])
                        tarck_y.append(res['ls'][-1, 1])
            if is_tarck:
                ax.scatter(tarck_x, tarck_y, c='firebrick', alpha=1,s=8,label='prediction')
                points = np.column_stack((tarck_x, tarck_y))
                ref_point = np.array([1.0, 1.0])
                #hv = get_performance_indicator("hv", ref_point=ref_point)
                #volume = hv.calc(points)
                #print("Hypervolume: ", volume)
            if method == 'NSGA3':
                res = nsga3_search(concave_fun_eval, rs, max_iters=300)
                generations = [300]
                for i,res_item in enumerate(res):
                    ax.scatter(res_item[:, 0], res_item[:, 1], s=6, c=nsga_colors[-1], alpha=1,label = r'n={}'.format(generations[i]))
            if method == 'MOBO':
                res,nds = mobo_search()
                print(nds.shape)
                ax.scatter(-res[:, 0], -res[:, 1], s=6, c='b', alpha=1)
                ax.scatter(-nds[:, 0], -nds[:, 1], s=6, c='r', alpha=1,label = 'non dominated')

            ax.set_xlabel(r'$l_1$')
            ax.set_ylabel(r'$l_2$')
            ax.xaxis.set_label_coords(1.015, -0.03)
            ax.yaxis.set_label_coords(-0.01, 1.01)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.grid(linestyle = '--', linewidth = 0.5)
            ax.legend(prop={'size':5,},loc = 3)
            fig.savefig('figures/' + method + '.pdf')

        plt.show()
