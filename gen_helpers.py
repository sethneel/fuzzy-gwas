import numpy as np
import pandas as pd
import itertools as it

# please use the following function (or something like it) to read files
def pname(name, path):
    '''Prepend the path to the filename'''
    return path + '/' + name

def popen(name):
    '''Open file in the path'''
    return open(pname(name))

# read in SNPs
def read_snp(file):
    '''Read a snp file into a pandas dataframe'''
    return(pd.read_table(
        file,
        sep='\s+', # columns are separated by whitespace
        # names of the columns
        names=[None, 'chromosome', 'morgans', 'position', 'ref', 'alt'],
        index_col=0
    ))


def read_snp_pop(pop):
    return(read_snp(pname(pop + '.snp')))


def get_chr_range(chromosome, SNPs):
    '''Returns the range of positions where SNPs for a chromosome are kept'''
    filt = SNPs.query('chromosome=={}'.format(chromosome))
    start = SNPs.index.get_loc(filt.iloc[0].name)
    stop  = SNPs.index.get_loc(filt.iloc[-1].name) + 1
    return(start, stop)


# read in individual data
def read_ind(file):
    '''Read an ind file into a pandas dataframe'''
    return pd.read_table(
        file,
        sep='\s+',
        names=[None, 'sex', 'pop'],
        index_col=0
    )

def read_ind_pop(pop):
    return(read_ind(pname(pop + '.ind')))


def read_geno(file):
    '''Reads a geno file into a numpy matrix'''
    return(np.genfromtxt(
        file,               # the file
        dtype='uint8',      # read the data in as 1-byte integers
        delimiter=1,        # 1-byte width data
        missing_values=9,   # 9 indicates missing data
        usemask=True        # return a masked array
    ))


def read_geno_pop(pop):
    return read_geno(pname(pop + '.geno'))


def read_geno_pop_chr(pop, chromosome):
    '''Reads a slice of a geno file into a numpy matrix'''
    f = popen(pop + '.geno')      # open the file
    (start, stop) = get_chr_range(chromosome)
    s = it.islice(f, start, stop) # slice the file
    return read_geno(s)

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False