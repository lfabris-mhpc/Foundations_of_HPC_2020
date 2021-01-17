import math as m
import numpy as np
import pandas as pd

def blur_cost(p_r, i, p_c, j, block_r, block_c, kradius):
    h_up = i != 0
    h_down = i != p_r - 1
    h_left = j != 0
    h_right = j != p_c - 1

    kdiam = 2 * kradius + 1
    q = (2 * kdiam * kradius - kradius * (kradius + 1)) // 2

    r1 = block_r
    c1 = block_c

    if not h_up:
        r1 -= kradius
    if not h_down:
        r1 -= kradius
    if not h_left:
        c1 -= kradius
    if not h_right:
        c1 -= kradius

    #don't worry if block is big enough for 2 * kradius
    cost = r1 * c1 * kdiam**2
    if h_up:
        cost += c1 * q
        if h_left:
            cost += q**2
        if h_right:
            cost += q**2
    if h_down:
        cost += c1 * q
        if h_left:
            cost += q**2
        if h_right:
            cost += q**2

    if h_left:
        cost += r1 * q
    if h_right:
        cost += r1 * q

    return cost

#to be defined
def bwidth_cpu(size):
    return 10 * 10**9

def bwidth_read_disk(size):
    return 1 * 10**9

def bwidth_write_disk(size):
    return 1 * 10**9

def tread_block(rows, cols, pxsize, lat_disk):
    size = rows * cols * pxsize
    return 2 * lat_disk + size / bwidth_read_disk(size)

def twrite_row(p_r, p_c, cols, pxsize, lat_cpu, lat_disk):
    size_c = pxsize * cols
    size_cb = pxsize * m.ceil(cols / p_c)

    #collect first rows (gather on proc group)
    t = 2 * lat_cpu + size_cb / bwidth_cpu(size_cb)
    t += 2 * lat_disk + size_c / bwidth_write_disk(size_c)
    return t

def twrite_blockrows(p_r, p_c, brows, cols, pxsize, lat_cpu, lat_disk):
    return brows * (2 * lat_disk + p_c * twrite_row(p_r, p_c, cols, pxsize, lat_cpu, lat_disk))

def twrite_img_collective(p_r, p_c, rows, cols, pxsize, lat_cpu, lat_disk):
    return p_r * twrite_blockrows(p_r, p_c, m.ceil(rows / p_r), cols, pxsize, lat_cpu, lat_disk)

def twrite_img_single(rows, cols, pxsize, lat_disk):
    size = rows * cols * pxsize
    return 2 * lat_disk + size / bwidth_write_disk(size)

"""
for i in range(8, 12):
    x = i * 1000
    y = i * 1000
    print(f"blur_cost {x}*{y} {blur_op * blur_cost(1, 0, 1, 0, x, y, kradius)}")
x = 10000
y = 12000
print(f"blur_cost {x}*{y} {blur_op * blur_cost(1, 0, 1, 0, x, y, kradius)}")
"""

def balanced_divisors(p):
    a = 1
    diff = p - 1
    for b in range(1, p):
        if (p % b) == 0:
            d = int(abs(b - p // b))
            if d < diff:
                diff = d
                a = b
    b = p // a
    return int(max(a, b)), int(min(a, b))

def domain_decomp(p_r, p_c, img_r, img_c, kradius):
    block_rows = np.zeros((p_r, p_c), dtype=np.int32)
    block_cols = np.zeros_like(block_rows)

    r = img_r // p_r
    c = img_c // p_c
    #print(f"r, c: {r}, {c}")

    for i in range(p_r):
        block_rows[i, :] = r

        if i < img_r % p_r:
            block_rows[i, :] += 1

    for i in range(p_c):
        block_cols[:, i] = c

        if i < img_c % p_c:
            block_cols[:, i] += 1

    #print(f"block_rows {block_rows}")
    #print(f"block_cols {block_cols}")

    haloed_rows = block_rows.copy()
    haloed_cols = block_cols.copy()

    for i in range(p_r-1):
        haloed_rows[i, :] += kradius
        haloed_rows[i+1, :] += kradius

    for i in range(p_c-1):
        haloed_cols[:, i] += kradius
        haloed_cols[:, i+1] += kradius

    #print(f"haloed_rows {haloed_rows}")
    #print(f"haloed_cols {haloed_cols}")

    return block_rows, block_cols, haloed_rows, haloed_cols

def scaling(p, img_r, img_c, kradius, strong):
    ht_factor = 1
    if p > 24:
        slowed = 2 * (p - 24)
        ht_factor = (p - slowed + slowed * 1 / 0.6) / p
        print(f"slowed by hyperthreading: {ht_factor}")

    p_r, p_c = balanced_divisors(p)
    #print(f"mesh[{p_r}, {p_c}]")

    block_rows, block_cols, haloed_rows, haloed_cols = domain_decomp(p_r, p_c, img_r, img_c, kradius)

    file_read_omp = file_read_base
    #file_read_mpi = haloed_rows * haloed_cols * file_read_base / (img_r * img_c)
    file_read_mpi = file_read_omp * np.ones_like(block_rows)
    if p_r > 1 and p_c > 1:
        for i in range(p_r):
            for j in range(p_c):
                file_read_mpi[i, j] = tread_block(block_rows[i, j], block_cols[i, j], pxsize, lat_disk)
    #print(f"file_read_mpi[{p_r}, {p_c}]: {np.max(file_read_mpi)}")

    preproc_omp = preproc_base * ht_factor / p
    preproc_mpi = preproc_base * ht_factor * np.ones_like(block_rows)
    if p_r > 1 and p_c > 1:
        for i in range(p_r):
            for j in range(p_c):
                preproc_mpi[i, j] = haloed_rows[i, j] * haloed_cols[i, j] * preproc_base / (img_r * img_c)
    #print(f"preproc_mpi[{p_r}, {p_c}]: {np.max(preproc_mpi)}")

    kernel_init_omp = kernel_init_base * ht_factor / p
    kernel_init_mpi = kernel_init_base * ht_factor * np.ones_like(block_rows)
    #print(f"kernel_init_mpi[{p_r}, {p_c}]: {np.max(kernel_init_mpi)}")

    blur_omp = blur_cost(p_r, 0, p_c, 0, img_r, img_c, kradius) * blur_op * ht_factor / p
    blur_mpi = blur_base * ht_factor * np.ones_like(block_rows)
    #blur[1:p_r-2, 1:p_c-2] = pblock_rows * pbloc_cols * blur_base / (blur_base_factor * img_r * img_c)
    for i in range(p_r):
        for j in range(p_c):
            blur_mpi[i, j] = blur_cost(p_r, i, p_c, j, block_rows[i, j], block_cols[i, j], kradius) * blur_op * ht_factor
            print(f"blur_mpi[{p_r}, {p_c}][{i}, {j}]: {blur_mpi[i, j]}")
    blur_perfect = blur_cost(1, 0, 1, 0, block_rows[0, 0], block_cols[0, 0], kradius)
    #print(f"blur_mpi[{p_r}, {p_c}]: {np.max(blur_mpi)} (perfect would be {blur_base / p})")

    #postproc = haloed_rows * haloed_cols * postproc_base / (img_r * img_c)
    postproc_omp = postproc_base * ht_factor / p
    postproc_mpi = postproc_base * ht_factor * np.ones_like(block_rows)
    if p_r > 1 and p_c > 1:
        for i in range(p_r):
            for j in range(p_c):
                postproc_mpi[i, j] = block_rows[i, j] * block_cols[i, j] * postproc_base / (img_r * img_c)
    #print(f"postproc_mpi[{p_r}, {p_c}]: {np.max(postproc_mpi)}")

    file_write_omp = file_write_base
    #file_write_mpi = block_rows * block_cols * file_write_base / (img_r * img_c)
    file_write_mpi = file_write_base
    if p_r > 1 and p_c > 1:
        file_write_mpi = twrite_img_collective(p_r, p_c, img_r, img_c, pxsize, lat_cpu, lat_disk)
    #print(f"file_write_mpi[{p_r}, {p_c}]: {np.max(file_write_mpi)}")

    twall_omp = file_read_omp + preproc_omp + kernel_init_omp + blur_omp + postproc_omp + file_write_omp
    twall_mpi = np.max(file_read_mpi + preproc_mpi + kernel_init_mpi+ blur_mpi + postproc_mpi) + np.max(file_write_mpi)

    print(f"twall_omp[{p}]: {twall_omp}")
    print(f"twall_mpi[{p}]: {twall_mpi}")

    if strong:
        speedup_omp = twall_base / twall_omp
        print(f"speedup_omp[{p}]: {speedup_omp} overhead: {p - speedup_omp}")
        speedup_mpi = twall_base / twall_mpi
        print(f"speedup_mpi[{p}]: {speedup_mpi} overhead: {p - speedup_mpi}")
    else:
        efficiency_omp = twall_base / twall_omp
        print(f"efficiency_omp[{p}]: {efficiency_omp} overhead: {1 - efficiency_omp}")
        efficiency_mpi = twall_base / twall_mpi
        print(f"efficiency_mpi[{p}]: {efficiency_mpi} overhead: {1 - efficiency_mpi}")

    return twall_omp, twall_mpi

pxsize = 2
kradius = 50

lat_disk = 130 * 10**-9
lat_cpu = 34 * 10**-9

#strong scaling image
img_r = 21600
img_c = 21600

file_read_base = 0.5
preproc_base = 0.09
kernel_init_base = 0.00005
blur_base = 5130
blur_op = blur_base / blur_cost(1, 0, 1, 0, img_r, img_c, kradius)

postproc_base = 0.1
file_write_base = 0.4
twall_base = file_read_base + preproc_base + kernel_init_base + blur_base + postproc_base + file_write_base
print(f"twall_base (strong scaling): {twall_base}")

omp_total = 0
mpi_total = 0
print(f"strong scaling")
ps = [1] + list(range(4, 49, 4))
for p in ps:
    twall_omp, twall_mpi = scaling(p, img_r, img_c, kradius, True)
    print("")

    if p <= 24:
        omp_total += twall_omp
    mpi_total += twall_mpi

print(f"omp_total: {omp_total} ({omp_total/3600} hours)")
print(f"mpi_total: {mpi_total} ({mpi_total/3600} hours)")
print("")


#weak scaling image
img_r = 10000
img_c = 12000

file_read_base = 0.5
preproc_base = 0.09
kernel_init_base = 0.00005
blur_base = 5130 / 3.5
blur_op = blur_base / blur_cost(1, 0, 1, 0, img_r, img_c, kradius)

postproc_base = 0.1
file_write_base = 0.4
twall_base = file_read_base + preproc_base + kernel_init_base + blur_base + postproc_base + file_write_base
print(f"twall_base (strong scaling): {twall_base}")

omp_total = 0
mpi_total = 0
print(f"weak scaling")
ps = [1] + list(range(4, 49, 4))
for p in ps:
    p_r, p_c = balanced_divisors(p)
    twall_omp, twall_mpi = scaling(p, img_r * p_r, img_c * p_c, kradius, False)
    print("")

    if p <= 24:
        omp_total += twall_omp
    mpi_total += twall_mpi

print(f"omp_total: {omp_total} ({omp_total/3600} hours)")
print(f"mpi_total: {mpi_total} ({mpi_total/3600} hours)")
