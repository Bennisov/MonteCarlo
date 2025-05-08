import numpy
import matplotlib.pyplot as plt
import numba
from scipy.stats import chi2


@numba.njit
def f(x):
    return (4/5) * (1 + x - x**3)


@numba.njit
def F(x):
    return (4/5) * (x + x**2/2 - x**4/4)


def plot_hist(X, title):
    plt.hist(X, bins=10, density=True, alpha=0.6, color='b', edgecolor='black')
    x = numpy.linspace(0, 1, 1000)
    plt.plot(x, f(x), 'r', lw=2)
    plt.title(title)
    plt.ylim(0.8, 1.15)
    plt.savefig(title + '.png')
    plt.show()


@numba.njit
def roz(N):
    U1 = numpy.random.uniform(0, 1, size=N)
    U2 = numpy.random.uniform(0, 1, size=N)
    X = numpy.where(U1 <= 4/5, U2, numpy.sqrt(1 - numpy.sqrt(1 - U2)))
    return X


@numba.njit
def marek(N, delta):
    X = numpy.zeros(N)
    X[0] = numpy.random.uniform(0, 1)
    for i in range(1, N):
        x_new = X[i - 1] + (2 * numpy.random.uniform(0, 1) - 1) * delta
        if 0 <= x_new <= 1:
            acc = f(x_new) / f(X[i - 1])
            if numpy.random.uniform(0, 1) <= min(1, acc):
                X[i] = x_new
            else:
                X[i] = X[i - 1]
        else:
            X[i] = X[i - 1]
    return X


@numba.njit
def elim(N):
    X = numpy.zeros(N) - 1
    k = 0
    while k < N:
        U1 = numpy.random.uniform(0, 1)
        G2 = 1.15 * numpy.random.uniform(0, 1)
        if G2 <= f(U1):
            X[k] = U1
            k += 1
    return X


N = int(1e6)
X_roz = roz(N)
X_mar_5 = marek(N, 0.5)
X_mar_05 = marek(N, 0.05)
X_eli = elim(N)
plot_hist(X_roz, 'Rozkład złożony')
plot_hist(X_mar_5, 'Łańcuch Markowa dx=0.5')
plot_hist(X_mar_05, 'Łańcuch Markowa dx=0.05')
plot_hist(X_eli, 'Metoda eliminacji')

X_roz_hist, freq = numpy.histogram(X_roz, bins=10)
X_mar_5_hist, _ = numpy.histogram(X_mar_5, bins=10)
X_mar_05_hist, _ = numpy.histogram(X_mar_05, bins=10)
X_eli_hist, _ = numpy.histogram(X_eli, bins=10)

x_bins = numpy.linspace(0, 1, 11)
sum_roz = 0
sum_mar_5 = 0
sum_mar_05 = 0
sum_eli = 0
for i in range(len(x_bins)-1):
    pi = F(x_bins[i+1]) - F(x_bins[i])
    sum_roz += (X_roz_hist[i] - pi * N)**2 / (pi * N)
    sum_mar_5 += (X_mar_5_hist[i] - pi * N)**2 / (pi * N)
    sum_mar_05 += (X_mar_05_hist[i] - pi * N)**2 / (pi * N)
    sum_eli += (X_eli_hist[i] - pi * N)**2 / (pi * N)
chi2_crit = chi2.ppf(0.95, df=9)
print(f"Chi2(deg=9, a=0.05): {chi2_crit}")
print(f"Chi2(rozkład złożony): {sum_roz},  Hipoteza: {sum_roz<chi2_crit}")
print(f"Chi2(Łańcuch Markowa, dx=0.5): {sum_mar_5},  Hipoteza: {sum_mar_5<chi2_crit}")
print(f"Chi2(Łańcuch Markowa, dx=0.05): {sum_mar_05},  Hipoteza: {sum_mar_05<chi2_crit}")
print(f"Chi2(Metoda eliminacji): {sum_eli},  Hipoteza: {sum_eli<chi2_crit}")