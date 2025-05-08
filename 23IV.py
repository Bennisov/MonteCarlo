import numpy as np
import matplotlib.pyplot as plt
import numba


k1 = 1.
k2 = 1.
k3 = .001
k4 = .01
tmax = 200
N = 50
Pmax = 100


#@numba.njit
def sim(k1, k2, k3, k4, x10, x20, x30, tmax, N, Pmax):
    t_grid = np.linspace(0, tmax, N)
    all_x1 = np.zeros((Pmax, N))
    all_x2 = np.zeros((Pmax, N))
    all_x3 = np.zeros((Pmax, N))
    for p in range(Pmax):
        t = 0.0
        x1, x2, x3 = x10, x20, x30
        t_series = []
        x1_series = []
        x2_series = []
        x3_series = []
        while t < tmax:
            l1 = k1
            l2 = k2
            l3 = k3 * x1 * x2
            l4 = k4 * x3
            l = [l1, l2, l3, l4]
            lmax = sum(l)
            if lmax == 0:
                break
            delta_t = -np.log(np.random.rand()) / lmax
            t += delta_t
            u = np.random.rand()
            thresholds = np.cumsum(l) / lmax
            if u < thresholds[0]:
                x1 += 1
            elif u < thresholds[1]:
                x2 += 1
            elif u < thresholds[2]:
                if x1 > 0 and x2 > 0:
                    x1 -= 1
                    x2 -= 1
                    x3 += 1
            else:
                if x3 > 0:
                    x3 -= 1
            t_series.append(t)
            x1_series.append(x1)
            x2_series.append(x2)
            x3_series.append(x3)
        all_x1[p] = np.interp(t_grid, t_series, x1_series)
        all_x2[p] = np.interp(t_grid, t_series, x2_series)
        all_x3[p] = np.interp(t_grid, t_series, x3_series)
    return t_grid, all_x1, all_x2, all_x3


t_grid, x1, x2, x3 = sim(k1=k1, k2=k2, k3=k3, k4=k4, x10=120, x20=80, x30=1, tmax=tmax, N=N, Pmax=Pmax)
plt.figure(figsize=(12, 6))
for i in range(Pmax):
    plt.plot(t_grid, x1[i], label=f'x1 - tra {i+1}', c='red')
    plt.plot(t_grid, x2[i], label=f'x2 - tra {i+1}', c='blue')
    plt.plot(t_grid, x3[i], label=f'x3 - tra {i+1}', c='green')
plt.xlabel('t')
plt.ylabel('n particles')
plt.title(f'Trajektorie x1, x2, x3(t) dla Pmax = {Pmax}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

x3_mean = np.mean(x3, axis=0)
x3_std = np.std(x3, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(t_grid, x3_mean)
plt.fill_between(t_grid, x3_mean - x3_std, x3_mean + x3_std, alpha=0.3)
plt.xlabel('t')
plt.ylabel('x3')
plt.title(f'Średnia x3(t) z odchyleniem standardowym (Pmax={Pmax})')
plt.grid()

plt.tight_layout()
plt.savefig('23IV_3.png')
plt.show()