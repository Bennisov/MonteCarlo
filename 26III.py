import numpy
import matplotlib.pyplot as plt
import numba


@numba.njit
def normal_distribution_2D(N):
    U1 = numpy.random.uniform(0, 1, size=N)
    U2 = numpy.random.uniform(0, 1, size=N)
    X = numpy.sqrt(-2 * numpy.log(1 - U1)) * numpy.cos(2 * numpy.pi * U2)
    Y = numpy.sqrt(-2 * numpy.log(1 - U1)) * numpy.sin(2 * numpy.pi * U2)
    return X, Y


@numba.njit
def transform_circle(X, Y, N):
    X1 = X / numpy.sqrt(X**2 + Y**2)
    Y1 = Y / numpy.sqrt(X**2 + Y**2)
    U1 = numpy.random.uniform(0, 1, size=N)
    X2 = numpy.sqrt(U1) * X1
    Y2 = numpy.sqrt(U1) * Y1
    return X1, Y1, X2, Y2


def transform_elipse(a, b1, b2, X, Y):
    s = numpy.sin(a)
    c = numpy.cos(a)
    Ra = numpy.array([[c, -s], [s, c]])
    r1 = b1 * numpy.matmul(Ra, numpy.array([[1], [0]]))
    r2 = b2 * numpy.matmul(Ra, numpy.array([[0], [1]]))
    A = numpy.hstack((r1, r2))
    X_vec = numpy.vstack((X, Y))
    Xp = numpy.matmul(A, X_vec)
    return Xp


def plot_distribution_2D(X, Y, title):
    plt.figure()
    plt.scatter(X, Y, s=0.5)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()


@numba.njit
def calc_cov(X, Y, N):
    x_av = numpy.sum(X) / N
    y_av = numpy.sum(Y) / N
    x2_av = numpy.sum(X**2) / N
    y2_ev = numpy.sum(Y**2) / N
    xy_av = numpy.sum(X*Y) / N
    sigma_x2 = x2_av - x_av**2
    sigma_y2 = y2_ev - y_av**2
    sigma_xy = xy_av - x_av*y_av
    rxy = sigma_xy / numpy.sqrt(sigma_x2 * sigma_y2)
    return numpy.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]]), rxy


N = 1e4
X, Y = normal_distribution_2D(N=int(N))
# plot_distribution_2D(X, Y, title='Normal distribution')
X1, Y1, X2, Y2 = transform_circle(X, Y, N=int(N))
# plot_distribution_2D(X1, Y1, title='Uniform distribution on circle')
# plot_distribution_2D(X2, Y2, title='Uniform distribution in circle')
a = numpy.pi/4
b1 = 1
b2 = 0.2
Xp = transform_elipse(a, b1, b2, X2, Y2)
#plot_distribution_2D(Xp[0], Xp[1], title='Uniform distribution in elipse')
cov, rxy = calc_cov(Xp[0], Xp[1], N)
print("Macierz kowariancji dla jednorodnego rozkładu na elipsie:")
print(cov)
print("Wspolczynnik korelacji dla jednorodnego rozkładu na elipsie:")
print(rxy)
Xp2 = transform_elipse(a, b1, b2, X, Y)
#plot_distribution_2D(Xp2[0], Xp2[1], title='Gauss in elipse')
cov, rxy = calc_cov(Xp2[0], Xp2[1], N)
print("Macierz kowariancji dla rozkładu Gaussa na elipsie:")
print(cov)
print("Wspolczynnik korelacji dla rozkładu Gaussa na elipsie:")
print(rxy)
