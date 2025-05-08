import numpy
import numba

R0 = 1.315  # A
R1 = 1.7  # A
R2 = 2.0  # A
De = 6.325  # eV
S = 1.29
Lam = 1.5  # A-1
Delta = 0.80469
A0 = 0.011304
C0 = 19
D0 = 2.5

atom_locations = numpy.loadtxt("atoms_positions_c60.dat", dtype=float)
xs, ys, zs = atom_locations[:, 0], atom_locations[:, 1], atom_locations[:, 2]


@numba.njit
def Vi_calc(xi, yi, zi, n, i, xs, ys, zs):
    Vi = 0.
    for j in range(n):
        if i == j:
            continue
        r = calc_dist_car(xi, yi, zi, xs[j], ys[j], zs[j])
        fcut = f_cut(r)
        if fcut == 0:
            continue
        b = (calc_b(i, j, n, xs, ys, zs) + calc_b(j, i, n, xs, ys, zs)) / 2
        Vi += f_cut(r) * (vr(r) - b * va(r))
    return Vi


@numba.njit
def calc_b(i, j, n, xs, ys, zs):
    xsi, ysi, zsi = xs[i], ys[i], zs[i]
    xsj, ysj, zsj = xs[j], ys[j], zs[j]
    ksi = 0
    for k in range(n):
        if (k == i) or (k == j):
            continue
        xsk, ysk, zsk = xs[k], ys[k], zs[k]
        r = calc_dist_car(xsi, ysi, zsi, xsk, ysk, zsk)
        fcut = f_cut(r)
        if fcut == 0:
            continue
        xij, yij, zij = xsj - xsi, ysj - ysi, zsj - zsi
        xik, yik, zik = xsk - xsi, ysk - ysi, zsk - zsi
        cos = dot_product(xij, yij, zij, xik, yik, zik) / (
                    calc_dist_car(xij, yij, zij, 0, 0, 0) * calc_dist_car(xik, yik, zik, 0, 0, 0))
        g = A0 * (1 + C0 ** 2 / D0 ** 2 - C0 ** 2 / (D0 ** 2 + (1 + cos) ** 2))
        ksi += fcut * g
    return numpy.power(1 + ksi, -Delta)


@numba.njit
def dot_product(x1, y1, z1, x2, y2, z2):
    return x1 * x2 + y1 * y2 + z1 * z2


@numba.njit
def vr(r):
    return De * numpy.exp(-numpy.sqrt(2 * S) * Lam * (r - R0)) / (S - 1)


@numba.njit
def va(r):
    return De * S * numpy.exp(-numpy.sqrt(2 / S) * Lam * (r - R0)) / (S - 1)


@numba.njit
def f_cut(r):
    if r > R2:
        return 0
    elif r > R1:
        return 1 / 2 * (1 + numpy.cos(numpy.pi * (r - R1) / (R2 - R1)))
    else:
        return 1


@numba.njit
def transf(r, theta, phi):
    return r * numpy.sin(theta) * numpy.cos(phi), r * numpy.sin(theta) * numpy.sin(phi), r * numpy.cos(theta)


@numba.njit
def calc_dist_sph(r1, theta1, phi1, r2, theta2, phi2):
    x1, y1, z1 = transf(r1, theta1, phi1)
    x2, y2, z2 = transf(r2, theta2, phi2)
    return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


@numba.njit
def calc_dist_car(x1, y1, z1, x2, y2, z2):
    return numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


@numba.njit
def calc_pcf(rs, thetas, phis, n):
    M = 100
    pcf = numpy.zeros(M)
    r_avg = numpy.sum(rs) / n
    r_max = 2.5 * r_avg
    dr = r_max / M
    for i in range(n):
        for j in range(i+1, n):
            r = calc_dist_sph(rs[i], thetas[i], phis[i], rs[j], thetas[j], phis[j])
            m = int(r / dr)
            if m<M: pcf[m] = pcf[m] + 2 * 4 * numpy.pi * r_avg**2 / (n**2 * 2 * numpy.pi * r * dr)


@numba.njit
def move_atoms(ri, thetai, phii, wr, wphi, wtheta, beta, i, coord):
    xs, ys, zs = transf(coord[0], coord[1], coord[2])
    U1 = numpy.random.random()
    U2 = numpy.random.random()
    U3 = numpy.random.random()
    dr = ri * (2*U1 - 1) * wr
    dphi = phii * (2*U2 - 1) * wphi
    dtheta = thetai * (2*U3 - 1) * wtheta
    ri_new = ri + dr
    thetai_new = thetai + dtheta
    phii_new = phii + dphi
    if phii_new < 0:
        phii_new = phii_new + 2 * numpy.pi
    if phii_new > 2 * numpy.pi:
        phii_new = phii_new - 2 * numpy.pi
    if (thetai_new < 0) or (thetai_new > numpy.pi):
        thetai_new = thetai
    x, y, z = transf(ri_new, thetai_new, phii_new)
    Vinew = Vi_calc(x, y, z, n, i, xs, ys, zs)
    x, y, z = transf(ri, thetai, phii)
    Viold = Vi_calc(x, y, z, n, i, xs, ys, zs)
    pacc = numpy.exp(-beta * (Vinew-Viold))
    if pacc > 1:
        pacc = 1
    U4 = numpy.random.random()
    if U4 > pacc:
        return ri, thetai, phii
    else:
        return ri_new, thetai_new, phii_new


@numba.njit
def sphere_change(coord, Wall, beta):
    rs = coord[0]
    U1 = numpy.random.random()
    r_new = numpy.zeros(n)
    for i in range(n):
        r_new[i] = rs[i] * (1 + Wall * (2 * U1 - 1))
    V_old = 0.
    V_new = 0.
    xs_old, ys_old, zs_old = transf(rs, coord[1], coord[2])
    xs_new, ys_new, zs_new = transf(r_new, coord[1], coord[2])
    for i in range(n):
        V_old += Vi_calc(xs_old[i], ys_old[i], zs_old[i], n, i)
        V_new += Vi_calc(xs_new[i], ys_new[i], zs_new[i], n, i)
    V_old, V_new = 1/2 * V_old, 1/2 * V_new
    pacc = numpy.exp(-beta * (V_new-V_old))
    if pacc > 1:
        pacc = 1
    U4 = numpy.random.random()
    if U4 > pacc:
        return rs
    else:
        return r_new


#z1
V_sum = 0.
n = int(xs.size)
for it in range(n):
    V_sum += Vi_calc(xs[it], ys[it], zs[it], n, it)
print(V_sum / 2)


#z2
n = 60
beta_min = 1
beta_max = 100
p = 2
it_max = int(1e5)
wr = 1e-4
wphi = .05
wtheta = .05
Wall = 1e-4
ri = 3.5
rs = numpy.array([ri] * n)
phis = 2 * numpy.pi * numpy.random.random(size=n)
thetas = numpy.pi * numpy.random.random(size=n)
for it in range(it_max):
    beta = beta_min + (it / it_max)**p * (beta_max - beta_min)
for i in range()