import numpy as np
cimport numpy as np
cimport cython

def get_acc(   np.ndarray[np.float64_t, ndim=2] pos,
            np.ndarray[np.float64_t, ndim=2] mass,
            double G,
            double softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
        pos  is an N x 3 matrix of positions
        mass is an N x 1 vector of masses
        G is Newton's Gravitational constant
        softening is the softening length
        a is N x 3 matrix of accelerations
    """

    cdef int N = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] x = pos[:, 0:1]
    cdef np.ndarray[np.float64_t, ndim=2] y = pos[:, 1:2]
    cdef np.ndarray[np.float64_t, ndim=2] z = pos[:, 2:3]

    cdef np.ndarray[np.float64_t, ndim=2] dx = x.T - x
    cdef np.ndarray[np.float64_t, ndim=2] dy = y.T - y
    cdef np.ndarray[np.float64_t, ndim=2] dz = z.T - z

    cdef np.ndarray[np.float64_t, ndim=2] inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    cdef np.ndarray[np.float64_t, ndim=2] ax = G * (dx * inv_r3) @ mass
    cdef np.ndarray[np.float64_t, ndim=2] ay = G * (dy * inv_r3) @ mass
    cdef np.ndarray[np.float64_t, ndim=2] az = G * (dz * inv_r3) @ mass

    cdef np.ndarray[np.float64_t, ndim=2] a = np.hstack((ax, ay, az))

    return a

def getEnergy(np.ndarray[np.float64_t, ndim=2] pos,
              np.ndarray[np.float64_t, ndim=2] vel, 
              np.ndarray[np.float64_t, ndim=2] mass, 
              double G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """

    cdef int N = pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] x = pos[:, 0:1]
    cdef np.ndarray[np.float64_t, ndim=2] y = pos[:, 1:2]
    cdef np.ndarray[np.float64_t, ndim=2] z = pos[:, 2:3]

    # Kinetic Energy:
    cdef double KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # Potential Energy:
    cdef np.ndarray[np.float64_t, ndim=2] dx = x.T - x
    cdef np.ndarray[np.float64_t, ndim=2] dy = y.T - y
    cdef np.ndarray[np.float64_t, ndim=2] dz = z.T - z

    cdef np.ndarray[np.float64_t, ndim=2] inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    cdef double PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE
