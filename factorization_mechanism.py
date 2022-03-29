
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
    w = [np.random.normal(0,1) for _ in range(m)]
    y = np.matmul(w, qmn)
    thresh = np.mean(y)
    y = [0 if u > thresh else 1 for u in y]


if __name__ == "__main__":
    # please set the path to your data directory here
    home_dir = '/Users/sneel/'
    data_path = home_dir + 'Dropbox/Research/Current_Projects/epi-data/Data'
    # read in genotype data from CEU population (112 respondents, 718848 snps)
    geno_matrix = gen_helpers.read_geno(gen_helpers.pname('CEU.geno', data_path))
    # remove SNPs with missing data: 500k by 112
    geno_matrix = np.ma.compress_rows(geno_matrix)
    # test on small matrix
    m = 500
    n = 100
    query_matrix = geno_matrix[0:m, 0:n]
    with open('query_submatrix.pickle', 'wb') as f:
        pickle.dump(query_matrix, f, pickle.HIGHEST_PROTOCOL)


    # start here if don't want to load the data
    with open('query_submatrix.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        query_matrix = pickle.load(f)

    m = 250
    n = 100
    query_matrix_mn = query_matrix[0:m, 0:n]
    constr_mat = create_constr_mat(query_matrix_mn)
    prob, x_variable = create_SDP(*constr_mat)
    prob.solve()
    eta_opt = prob.value
    x_opt = x_variable.value[0:(m+n), 0:(m+n)]
    L, R = get_matrix_decomp(x_opt, m, n, W=query_matrix_mn)
    R_norm = np.max(np.linalg.norm(R, axis=1))
    L_norm = np.max(np.linalg.norm(L, axis=1))
    #note R_norm * L_norm = eta_opt
    print(eta_opt)