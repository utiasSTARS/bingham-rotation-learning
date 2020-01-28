import numpy as np

def wigner_log_likelihood_measure(A, reduce=False):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    lls = np.log(spacings) - 0.25*np.pi*(spacings**2)
    if reduce:
        return np.sum(lls, axis=1).mean()
    else:
        return np.sum(lls, axis=1)


def first_eig_gap(A):
    el = np.linalg.eigvalsh(A)
    spacings = np.diff(el, axis=1)
    return spacings[:, 0] 

def det_inertia_mat(A):
    #A_inertia = -A
    
    els = np.linalg.eigvalsh(A)

    els = els[:, 1:] - els[:, 0, None] 
    # min_el = els[:,0]
    # I = np.repeat(np.eye(4).reshape(1,4,4), A_inertia.shape[0], axis=0)
    # A_inertia = A_inertia + I*min_el[:,None,None]

    return els[:,0]*els[:,1]*els[:,2] #np.linalg.det(A_inertia)

def sum_bingham_dispersion_coeff(A):
    if len(A.shape) == 2:
        A = A.reshape(1,4,4)
    els = np.linalg.eigvalsh(A)
    min_el = els[:,0]
    I = np.repeat(np.eye(4).reshape(1,4,4), A.shape[0], axis=0)
    return np.trace(-A + I*min_el[:,None,None], axis1=1, axis2=2)

   
def l2_norm(vecs):
    return np.linalg.norm(vecs, axis=1)

#Used for autoencoder
def l1_norm(l1_means):
    return l1_means

def decode_metric_name(uncertainty_metric_fn):
    if uncertainty_metric_fn == first_eig_gap:
        return 'First Eigenvalue Gap'
    elif uncertainty_metric_fn == sum_bingham_dispersion_coeff:
        return 'tr($\mathbf{\Lambda}$)'
    elif uncertainty_metric_fn == det_inertia_mat:
        return 'Det of Inertia Matrix (min eigvalue added)'
    elif uncertainty_metric_fn == l1_norm:
        return 'Average $L_1$ reconstruction error'
    else:
        raise ValueError('Unknown uncertainty metric')

def compute_threshold(A, uncertainty_metric_fn=first_eig_gap, quantile=0.75):
    #stats = wigner_log_likelihood(A)
    stats = uncertainty_metric_fn(A)
    return np.quantile(stats, quantile)

def compute_mask(A, uncertainty_metric_fn, thresh):
    if uncertainty_metric_fn == first_eig_gap:
        return uncertainty_metric_fn(A) > thresh
    elif uncertainty_metric_fn == sum_bingham_dispersion_coeff:
        return uncertainty_metric_fn(A) < thresh
    elif uncertainty_metric_fn == l2_norm:
        return uncertainty_metric_fn(A) > thresh
    elif uncertainty_metric_fn == l1_norm:
        return uncertainty_metric_fn(A) < thresh
        
    else:
        raise ValueError('Unknown uncertainty metric')