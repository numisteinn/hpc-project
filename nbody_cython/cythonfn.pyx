# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt, pow

cnp.import_array()

DTYPE = np.float64

ctypedef cnp.float64_t DTYPE_t

def get_acc(
    cnp.ndarray[DTYPE_t, ndim=2] pos,
    cnp.ndarray[DTYPE_t, ndim=1] mass,
    double G,
    double softening
):

    cdef int N = pos.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] acc = np.zeros((N, 3), dtype=DTYPE)
    cdef int i, j
    cdef double dx, dy, dz, inv_r3, r2

    # Compute pairwise accelerations
    for i in prange(N, nogil="True"):
        for j in range(N):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]

                r2 = dx * dx + dy * dy + dz * dz + softening * softening
                inv_r3 = pow(r2, -1.5) if r2 > 0 else 0.0

                acc[i, 0] += G * dx * inv_r3 * mass[j]
                acc[i, 1] += G * dy * inv_r3 * mass[j]
                acc[i, 2] += G * dz * inv_r3 * mass[j]

    return acc


def get_energy(
    cnp.ndarray[DTYPE_t, ndim=2] pos,
    cnp.ndarray[DTYPE_t, ndim=2] vel,
    cnp.ndarray[DTYPE_t, ndim=1] mass,
    double G
):
    cdef int N = pos.shape[0]
    cdef int i, j
    cdef double KE = 0, PE = 0
    cdef double dx, dy, dz, inv_r

    # Compute kinetic energy (vectorized)
    for i in prange(N, nogil=True):
        KE += 0.5 * mass[i] * (vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2)

    # Compute potential energy
    for i in prange(N, nogil=True):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]

            inv_r = sqrt(dx * dx + dy * dy + dz * dz)
            inv_r = 1.0 / inv_r if inv_r > 0 else 0

            PE += -G * mass[i] * mass[j] * inv_r

    return KE, PE

