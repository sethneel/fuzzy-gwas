
import cvxpy
import scipy
import gen_helpers
import pickle
import numpy as np


def gaussian_mechanism(L, R, y, epsilon, delta):

    # Gaussian Mechanism
    # R^Ty + eta
    n = len(y)
    delta_r = np.max(np.linalg.norm(R, axis=1))/n
    sigma = np.sqrt(2*np.log(1.25/delta)*np.power(delta_r, 2)/np.power(epsilon, 2))
    # generate noise
    eta = np.random.normal(0, sigma, size=R.shape[1])

    # return the private and non-private correlations
    non_private = 1.0/n*np.matmul(L, np.matmul(np.transpose(R), y))
    return {'non_private': non_private, 'private': non_private + np.matmul(L, eta)}


def build_sparse_matrix(shape_list, non_zero_entries_list):
    a = scipy.sparse.dok_matrix(shape_list)
    for entry in non_zero_entries_list:
        a[entry[0][0], entry[0][1]] = entry[1]
    return a


def create_constr_mat(query_matrix):
    m, n = query_matrix.shape

    # we create three types of matrices, E_ij (BC^T = Q), P_i (X_ii > n), C (minimize eta)
    P_i = [build_sparse_matrix((m + n + 1, m + n + 1),
                               [[[i, i], 1], [[m + n, m + n], -1]]) for i in range(m+n)]
    E_ij = [(build_sparse_matrix((m + n + 1, m + n + 1), [[(j, m+i), .5], [(m+i, j), .5]]), query_matrix[j, i])
            for i in range(n) for j in range(m)]
    # zero out row/col m + n + 1 except diagonal
    Z_mn = [build_sparse_matrix((m + n + 1, m + n + 1), [[(m+n, i), 1]]) for i in range(m+n-1)]

    return [m, n, P_i, E_ij, Z_mn]


def create_SDP(m, n, P_i, E_ij, Z_mn):
    x = cvxpy.Variable((m+n+1, m+n+1), PSD=True)
    constraints = [x >> 0]
    constraints += [cvxpy.trace(P @ x) <= 0 for P in P_i]
    constraints += [cvxpy.trace(E[0] @ x) == E[1] for E in E_ij]
    constraints += [cvxpy.trace(Z @ x) == 0 for Z in Z_mn]
    c = build_sparse_matrix((m+n+1, m+n+1), [[[m+n, m+n], 1]])
    obj = cvxpy.Minimize(cvxpy.trace(c @ x))
    problem = cvxpy.Problem(obj, constraints)
    return problem, x


def get_matrix_decomp(x_opt, m, n, W=None, EPS=1e-8):
    U = np.linalg.cholesky(gen_helpers.nearestPD(x_opt))
    L = U[0:m, :]
    R = U[m:, :]
    if W is not None:
        max_diff = np.linalg.norm(W - np.matmul(L, np.transpose(R)), ord=np.inf)
        if max_diff > 1e-3:
            raise ValueError('matrix decomposition incorrect ||W - LR^T|| = {}'.format(max_diff))
    return L, R


def ldp_mechanism(query_matrix, y, epsilon, delta):
    # Gaussian Mechanism
    # Q(y + eta)
    n = len(y)
    delta_r = np.max(y)
    sigma = np.sqrt(2*np.log(1.25/delta)*np.power(delta_r, 2)/np.power(epsilon, 2))
    # generate noise
    eta = np.random.normal(0, sigma, size=n)

    # return the private and non-private correlations
    non_private = 1.0/n*np.matmul(query_matrix, y)
    return {'non_private': non_private, 'private': 1.0/n*np.matmul(query_matrix, y + eta)}


def generate_y(qmn):
    # generate binary y with some correlation
    m, n = qmn.shape
    w = [np.random.normal(0,1) for _ in range(m)]
    y = np.matmul(w, qmn)
    thresh = np.mean(y)
    y = [0 if u > thresh else 1 for u in y]
    return y

def factorization_mechanism(query_matrix_mn, save=True):
    constr_mat = create_constr_mat(query_matrix_mn)
    prob, x_variable = create_SDP(*constr_mat)
    m, n = query_matrix_mn.shape
    prob.solve(solver=cvxpy.CVXOPT)
    x_opt = x_variable.value[0:(m+n), 0:(m+n)]
    L, R = get_matrix_decomp(x_opt, m, n, W=query_matrix_mn)
    R_norm = np.max(np.linalg.norm(R, axis=1))
    L_norm = np.max(np.linalg.norm(L, axis=1))
    eta_opt = R_norm * L_norm
    np.max(np.linalg.norm(query_matrix_mn, axis=0))
    print('error like {} instead of {} from gaussian mechanism'.format(eta_opt/n, np.max(np.linalg.norm(query_matrix_mn, axis=0))))
    results = {'query_matrix': query_matrix_mn, 'L': L, 'R': R, 'x_opt': x_opt, 'eta_opt': eta_opt}
    # save results
    if save:
        with open('Results/sdp_results_m_{}_n_{}.pickle'.format(m, n), 'wb') as g:
            pickle.dump(results, g, pickle.HIGHEST_PROTOCOL)
    else:
        return results

if __name__ == "__main__":
    with open('Data/query_submatrix.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        query_matrix = pickle.load(f)

    m = 10
    n = 100
    query_matrix_mn = query_matrix[0:m, 0:n]
    constr_mat = create_constr_mat(query_matrix_mn)
    prob, x_variable = create_SDP(*constr_mat)
    prob.solve(solver=cvxpy.CVXOPT)
    eta_opt = prob.value
    x_opt = x_variable.value[0:(m+n), 0:(m+n)]
    L, R = get_matrix_decomp(x_opt, m, n, W=query_matrix_mn)
    R_norm = np.max(np.linalg.norm(R, axis=1))
    L_norm = np.max(np.linalg.norm(L, axis=1))
    #note R_norm * L_norm = eta_opt
    print(eta_opt)