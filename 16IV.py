from numba import njit
import numpy as np
import matplotlib.pyplot as plt


@njit
def simulate_wiener(N_particles, D, dt, N_steps):
    sigma = np.sqrt(2 * D * dt)
    x = np.zeros((N_steps + 1, N_particles))
    y = np.zeros((N_steps + 1, N_particles))

    for t in range(1, N_steps + 1):
        dx = np.random.normal(0, sigma, size=N_particles)
        dy = np.random.normal(0, sigma, size=N_particles)
        x[t] = x[t-1] + dx
        y[t] = y[t-1] + dy

    return x, y

def compute_diffusion_coeffs(x, y, dt):
    t_vals = np.arange(x.shape[0]) * dt
    Dxx = (np.mean(x**2, axis=1) - np.mean(x, axis=1)**2) / (2 * t_vals + 1e-10)
    Dyy = (np.mean(y**2, axis=1) - np.mean(y, axis=1)**2) / (2 * t_vals + 1e-10)
    Dxy = (np.mean(x*y, axis=1) - np.mean(x, axis=1)*np.mean(y, axis=1)) / (2 * t_vals + 1e-10)
    return t_vals, Dxx, Dyy, Dxy


def simulate_and_plot_wiener(N_list):
    D = 1.0
    dt = 0.1
    tmax = 100
    N_steps = int(tmax / dt)

    for N in N_list:
        x, y = simulate_wiener(N, D, dt, N_steps)
        t_vals, Dxx, Dyy, Dxy = compute_diffusion_coeffs(x, y, dt)

        plt.figure(figsize=(8,5))
        plt.plot(t_vals, Dxx, label="Dxx")
        plt.plot(t_vals, Dyy, label="Dyy")
        plt.plot(t_vals, Dxy, label="Dxy")
        plt.xlabel("t")
        plt.ylabel("Współczynnik dyfuzji")
        plt.title(f"N = {N}")
        plt.legend()
        plt.grid()
        plt.show()

        Nt = tmax / dt
        Dxx_avg = np.mean(Dxx)
        Dyy_avg = np.mean(Dyy)
        Dxy_avg = np.mean(Dxy)
        Dxx2_avg = np.mean(Dxx ** 2)
        Dyy2_avg = np.mean(Dyy ** 2)
        Dxy2_avg = np.mean(Dxy ** 2)
        sigma_Dxx = np.sqrt((Dxx2_avg - Dxx_avg ** 2) / Nt)
        sigma_Dyy = np.sqrt((Dyy2_avg - Dyy_avg ** 2) / Nt)
        sigma_Dxy = np.sqrt((Dxy2_avg - Dxy_avg ** 2) / Nt)
        print(f"N={N}: Dxx={Dxx_avg:.4f} +- {sigma_Dxx:.4}, Dyy={Dyy_avg:.4f} +- {sigma_Dyy:.4}, Dxy={Dxy_avg:.4f} +- {sigma_Dxy:.4}")


#simulate_and_plot_wiener([100, 1000, 10000, 100000])


