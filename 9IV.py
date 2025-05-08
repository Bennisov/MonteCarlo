import numpy
import numba
import matplotlib.pyplot as plt
import pandas


@numba.njit
def met_pod(g, a, b, n):
    x = numpy.random.uniform(a, b, n)
    g_avg = numpy.sum(g(x)) * (b - a) / n
    g2_avg = numpy.sum(g(x) ** 2) * (b - a) ** 2 / n
    sigma2_avg = abs(g2_avg - g_avg ** 2) / n
    return g_avg, numpy.sqrt(sigma2_avg), x


# @numba.njit
def met_los_sym(g, a, b, n, m, var=False):
    dx = (b - a) / m
    pm = 1 / m
    nm = int(pm * n)
    g_avg = 0
    sigma2_avg = 0
    xs = numpy.zeros(0)
    if var:
        sigmasm = numpy.zeros(m)
    for i in range(1, m + 1):
        down = a + dx * (i - 1)
        up = down + dx
        xim = numpy.random.uniform(down, up, nm)
        xs = numpy.append(xs, xim, axis=0)
        gm = numpy.sum(g(xim)) * (b - a) / nm
        g2m = numpy.sum(g(xim) ** 2) * (b - a) ** 2 / nm
        g_avg += gm
        if var:
            sigmasm[i - 1] = abs(g2m - gm ** 2)
        sigma2_avg += abs(g2m - gm ** 2)
    g_avg = g_avg * pm
    sigma2_avg = sigma2_avg * pm ** 2 / nm
    if var:
        return numpy.sqrt(sigmasm), numpy.sqrt(sigma2_avg)
    return g_avg, numpy.sqrt(sigma2_avg), xs


# @numba.njit
def met_los_war(g, a, b, n, m):
    if n > 1e2:
        n_sym = int(1e3)
    else:
        n_sym = int(1e2)
    xs = numpy.zeros(0)
    dx = (b - a) / m
    pm = 1 / m
    sigmasm, _ = met_los_sym(g, a, b, n_sym, m, var=True)
    g_avg = 0
    sigma2_avg = 0
    for i in range(1, m + 1):
        weights = sigmasm / sigmasm.sum()
        nm = int(n * weights[i - 1])
        nm = max(nm, 1)
        down = a + dx * (i - 1)
        up = down + dx
        xim = numpy.random.uniform(down, up, nm)
        xs = numpy.append(xs, xim, axis=0)
        gm = numpy.sum(g(xim)) * (b - a) / nm
        g2m = numpy.sum(g(xim) ** 2) * (b - a) ** 2 / nm
        g_avg += gm
        sigma2_avg += abs(g2m - gm ** 2) / nm
    g_avg = g_avg * pm
    sigma2_avg = sigma2_avg * pm ** 2
    return g_avg, numpy.sqrt(sigma2_avg), xs


@numba.njit
def g1(x):
    return 1 + numpy.tanh(x)


@numba.njit
def g2(x):
    return 1 / (1 + x ** 2)


@numba.njit
def g3(x):
    return numpy.cos(numpy.pi * x) ** 10


Ns = numpy.array([1e2, 1e3, 1e4, 1e5], dtype='int')
gs = [g1, g2, g3]
ejs = [-3, 0, 0]
bis = [3, 10, 1]
cs = numpy.zeros((4, 3, 3))
sigmas = numpy.zeros((4, 3, 3))
xs = [[[] for _ in range(3)] for _ in range(3)]
for N in range(4):
    for dzi in range(3):
        kwargs = {'g': gs[dzi], 'a': ejs[dzi], 'b': bis[dzi], 'n': Ns[N]}
        if N == 3:
            cs[N, dzi, 0], sigmas[N, dzi, 0], xs[dzi][0] = met_pod(**kwargs)
            cs[N, dzi, 1], sigmas[N, dzi, 1], xs[dzi][1] = met_los_sym(**kwargs, m=10)
            cs[N, dzi, 2], sigmas[N, dzi, 2], xs[dzi][2] = met_los_war(**kwargs, m=10)
        cs[N, dzi, 0], sigmas[N, dzi, 0], _ = met_pod(**kwargs)
        cs[N, dzi, 1], sigmas[N, dzi, 1], _ = met_los_sym(**kwargs, m=10)
        cs[N, dzi, 2], sigmas[N, dzi, 2], _ = met_los_war(**kwargs, m=10)
xs = numpy.array([[numpy.array(xm) for xm in xg] for xg in xs], dtype=object)
gs_labels = ['g1(x) = 1 + tanh(x)', 'g2(x) = 1 / (1 + x²)', 'g3(x) = cos^10(πx)']
methods = ['Podstawowa','Warstwowe nieoptymalne', 'Warstwowe optymalne']
for i in range(3):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for j in range(3):  # dla każdej metody
        counts, bins = numpy.histogram(xs[i, j], bins=10)
        axs[j].plot((bins[1:] + bins[:-1])/2, counts)
        axs[j].set_title(f'{methods[j]}')
        axs[j].set_ylim([0, counts.max()*1.1])
        axs[j].set_xlabel('x')
        axs[j].set_ylabel('Liczność')
    fig.suptitle(f'Histogramy dla {gs_labels[i]}')
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{i}.png')


for i in range(3):
    df = pandas.DataFrame({
        'N': Ns,
        'C_podst': cs[:, i, 0],
        'σ_podst': sigmas[:, i, 0],
        'R_podst [%]': 100 * sigmas[:, i, 0] / cs[:, i, 0],
        'C_sys': cs[:, i, 1],
        'σ_sys': sigmas[:, i, 1],
        'R_sys [%]': 100 * sigmas[:, i, 1] / cs[:, i, 1],
        'C_warstw': cs[:, i, 2],
        'σ_warstw': sigmas[:, i, 2],
        'R_warstw [%]': 100 * sigmas[:, i, 2] / cs[:, i, 2],
    })
    print(df.to_latex(index=False, float_format="%.6f"))

