from factorization_mechanism import *

# please set the path to your data directory here
home_dir = '/Users/sneel/'
data_path = home_dir + 'Dropbox/Research/Current_Projects/epi-data/Data'
# read in genotype data from CEU population (112 respondents, 718848 snps)
geno_matrix = gen_helpers.read_geno(gen_helpers.pname('CEU.geno', data_path))
# remove SNPs with missing data: 500k by 112
geno_matrix = np.ma.compress_rows(geno_matrix)
# test on small matrix
m = 1000
n = 100
query_matrix = geno_matrix[0:m, 0:n]
with open('query_submatrix.pickle', 'wb') as f:
    pickle.dump(query_matrix, f, pickle.HIGHEST_PROTOCOL)
# start here if don't want to load the data
with open('query_submatrix.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    query_matrix = pickle.load(f)
query_matrix_mn = query_matrix[0:m, 0:n]

# compute matrix decomposition by solving SDP
factorization_mechanism(query_matrix_mn, save=True)

# start here to load prior results
m = 1
n = 100
with open('sdp_results_m_{}_n_{}.pickle'.format(m, n), 'rb') as g:
    results = pickle.load(g)
L = results['L']
R = results['R']
query_matrix_mn = results['query_matrix'][0:m, 0:n]
y = generate_y(query_matrix_mn)

epsilon = 10.0
output = gaussian_mechanism(L, R, y, epsilon=epsilon, delta=1 / n)
baseline_ldp = ldp_mechanism(query_matrix_mn, y, epsilon=epsilon, delta=1 / n)
pyplot.scatter(output['non_private'], output['private'], label='noisy values')
pyplot.plot(output['non_private'], output['non_private'], label='perfect accuracy')
pyplot.ylabel('noisy values')
pyplot.xlabel('non-private values')
pyplot.title('private vs non private values, m = {}, n = {}, epsilon = {}'.format(m, n, epsilon))
pyplot.scatter(output['non_private'], baseline_ldp['private'], label='ldp_baseline')
pyplot.show()
pyplot.show()

# NOTE something weird going on with LDP values not being centered.
np.mean(np.power(output['non_private'] - output['private'], 2))
np.mean(np.power(output['non_private'] - baseline_ldp['private'], 2))
