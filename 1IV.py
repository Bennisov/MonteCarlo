import numpy
import numba
import matplotlib.pyplot as plt


@numba.njit
def gen_2D(xa, ya, ra, N):
    u1 = numpy.random.uniform(0, 1, N)
    u2 = numpy.random.uniform(0, 1, N)
    u3 = numpy.sqrt(numpy.random.uniform(0, 1, N))
    x = numpy.sqrt(-2 * numpy.log(1 - u1)) * numpy.cos(2 * numpy.pi * u2)
    y = numpy.sqrt(-2 * numpy.log(1 - u1)) * numpy.sin(2 * numpy.pi * u2)
    r = numpy.sqrt(x**2 + y**2)
    x = x/r
    y = y/r
    x = u3 * x * ra + xa
    y = u3 * y * ra + ya
    return x, y


@numba.njit
def check(x, y, xa, ya, ra):
    return (x - xa)**2 + (y - ya)**2 <= ra**2


@numba.njit
def common_area(xa, ya, ra, xb, yb, rb, N):
    x, y = gen_2D(xa, ya, ra, N)
    count = check(x, y, xb, yb, rb)
    count = numpy.sum(count)
    mu = count * numpy.pi * ra * ra / N
    mu2 = mu * numpy.pi * ra * ra
    sigma = numpy.sqrt((mu2 - mu**2) / N)
    return mu, sigma


def plot_circles(xa, ya, ra, xb, yb, rb, N):
    xA, yA = gen_2D(xa, ya, ra, N)
    xB, yB = gen_2D(xb, yb, rb, N)

    plt.figure(figsize=(6, 6))
    plt.scatter(xA, yA, s=1, color='blue', label='K_A')
    plt.scatter(xB, yB, s=1, color='red', label='K_B', alpha=0.5)

    circle_A = plt.Circle((xa, ya), ra, color='blue', fill=False)
    circle_B = plt.Circle((xb, yb), rb, color='red', fill=False)

    plt.gca().add_patch(circle_A)
    plt.gca().add_patch(circle_B)

    plt.axis("equal")
    plt.legend()
    plt.savefig('2IV_1.png')
    plt.show()


Ra = 1
Rb = numpy.sqrt(2) * Ra
xa = Ra + Rb
mu1, sigma1 = common_area(xa, 0 ,Ra, 0, 0, Rb, int(1e4))
print("mu1 = ", mu1, "sigma1 = ", sigma1)
plot_circles(xa, 0, Ra, 0, 0, Rb, int(1e4))

a = ['A', 'A', 'B', 'B']
Ra = 1
Rb = numpy.sqrt(2) * Ra
xas = [Rb + 0.5 * Ra, 0, Rb + 0.5 * Ra, 0]
mus = numpy.zeros((4,5))
sigmas = numpy.zeros((4,5))
for i in range(4):
    xa = xas[i]
    for j in range(5):
        n = int(10**(j+2))
        if a[i] == 'A':
            mus[i, j], sigmas[i, j] = common_area(xa, 0, Ra, 0, 0, Rb, n)
        else:
            mus[i, j], sigmas[i, j] = common_area(0, 0, Rb, xa, 0, Ra, n)
plt.figure()
it = [1e2, 1e3, 1e4, 1e5, 1e6]
plt.errorbar(it, mus[0, :], yerr=sigmas[0, :], c='black', fmt='o', label='A', alpha=0.7)
plt.errorbar(it, mus[2, :], yerr=sigmas[2, :], c='red', fmt='o', label='B', alpha=0.7)
plt.xscale('log')
plt.xlabel('Iteration')
plt.ylabel('Common area')
plt.title('$x_a = R_b + 0.5 * R_a$')
plt.legend(loc='best')
plt.savefig('2IV_2.png')
plt.show()

plt.figure()
it = [1e2, 1e3, 1e4, 1e5, 1e6]
plt.errorbar(it, mus[1, :], yerr=sigmas[1, :], c='black', fmt='o', label='A', alpha=0.7)
plt.errorbar(it, mus[3, :], yerr=sigmas[3, :], c='red', fmt='o', label='B', alpha=0.7)
plt.xscale('log')
plt.xlabel('Iteration')
plt.ylabel('Common area')
plt.title('$x_a = 0$')
plt.legend(loc='best')
plt.savefig('2IV_3.png')
plt.show()