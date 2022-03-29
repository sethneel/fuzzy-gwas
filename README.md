# Set Up 

[Data](https://www.dropbox.com/sh/mkii6omsn0lu80c/AAB5A2Ae1fdk7jk8KTJL5jBfa?dl=0https://www.sanger.ac.uk/resources/downloads/human/hapmap3.html) folder with data from the [HapMap3]() project. Create a `Results` folder in the same directory.  A preprocessed pandas matrix of CEU.geno is saved as `Data/query_matrix.pickle` on which experiments can be run. 

- In `factorization_mechanism.py` We implement the factorization mechanism of [Lower Bounds in Communication Complexity Based on
Factorization Norms](https://www.cs.huji.ac.il/~nati/PAPERS/ccfn.pdf) using `cvxpy` as the SDP solver. For the `l_infty` error over queries this is shown to be optimal in [The Power of Factorization Mechanisms in Local and Central Differential Privacy](https://arxiv.org/abs/1911.08339). 
- `fac_mech_experiments.py` compares this method to the LDP (adding noise to the y's) baseline, and to the vanilla Gaussian mechanism. 

Package installation: `pip install -r /path/to/requirements.txt`
