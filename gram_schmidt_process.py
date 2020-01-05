import numpy as np

def project(u, v):
    """Project vector u onto vector v."""
    return u.dot(v)*v/(np.linalg.norm(v))


def modified_gram_schmidt(v):
    """Returns a NxN matrix with rows that span the plane defined by normal vector v.
       These vectors are not normalized.
    """
    N = np.max(v.shape)
    V_out = np.zeros((N+1, N))
    V_out[0, :] = v
    for idx in range(N):
        u = np.zeros(N)
        u[idx] = 1
        for jdx in range(idx+1):
            u = u - project(u, V_out[jdx, :])
        V_out[idx+1, :] = u
    return V_out[1:, :]


def householder(v, ind=0):
    """Returns a NxN matrix with rows that span the plane defined by normal vector v.
    Uses the Householder reflection.

    This establishes continuity for the rotation matrix case, but not for the quaternion (yet).
    """
    N = np.max(v.shape)
    w = v.copy() # Why WAS ONLY negative working?
    w[ind] = w[ind] + np.linalg.norm(v)
    # Q = (w.dot(w))*np.eye(N) - 2*np.outer(w, w)
    # This proper Householder form is needed for double cover invariance (unit quaternions)
    # Thus, we can't avoid the problem at w = 0 (boooo)
    Q = np.eye(N) - 2 * np.outer(w, w)/(w.dot(w))
    return Q

def householder_simple(v):
    """Returns a NxN matrix with rows that span the plane defined by normal vector v.
    Uses the Householder reflection.

    This establishes continuity for the rotation matrix case, but not for the quaternion (yet).
    """
    N = np.max(v.shape)
    Q = np.eye(N) - 2*np.outer(v, v)
    return Q

if __name__ == '__main__':

    N_runs = 10000
    ind = 0
    max_check = np.zeros(N_runs)
    rank_check = np.zeros(N_runs)
    double_cover_invariance_check = np.zeros(N_runs)
    for idx in range(0, N_runs):
        a = np.random.rand(4)*1.0 - 2.0
        V1 = householder(a, ind=ind)
        V2 = householder(-a, ind=ind)
        # V1 = householder_simple(a)
        a_check1 = V1.dot(a)
        # a_check1 = a_check1[1:]
        a_check1 = np.delete(a_check1, ind)
        max_check[idx] = np.max(np.abs(a_check1))
        rank_check[idx] = np.min(np.abs(np.linalg.eigvals(V1)))
        V1[ind, :] = 0.
        V2[ind, :] = 0.
        VV1 = V1.T.dot(V1)
        VV2 = V2.T.dot(V2)
        double_cover_invariance_check[idx] = np.max(np.abs(VV1-VV2))
    print("Max deviation (should be 0): {:}".format(np.max(max_check)))
    print("Min eigenvalue (should be 1): {:}".format(np.min(rank_check)))
    print("Double cover invariance check (should be 0): {:}".format(np.max(double_cover_invariance_check)))


