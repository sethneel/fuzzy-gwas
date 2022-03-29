
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from factorization_mechanism import gaussian_mech_noise
import pdb

with open('Data/query_submatrix.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    query_matrix = pickle.load(f)


class SNPDataset(Dataset):
    def __init__(self, query_matrix):
        self.query_matrix = torch.tensor(query_matrix)
    def __len__(self):
        return query_matrix.shape[0]
    def __getitem__(self, idx):
        return query_matrix[idx, :], idx


# Goal: Factor query_matrix = LR^T such that LR^T ~ query_matrix and
# ||R||_1 and ||L||_1 are low (low row norm for L, R)

# first attempt: two separate variables L, R, optimize using alternating gradient descent
# (if failed, can try and parameterize L, R)
# optimize the function ||LR^{T}-Q||_1 + lambda*||R||_1*||L|
class Factorization(torch.nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, m, n, k, eps, delta):
        super(Factorization, self).__init__()
        # initialize weights with random numbers
        weights_L = torch.distributions.Uniform(0, 0.1).sample((m, k))
        weights_R = torch.distributions.Uniform(0, 0.1).sample((n, k))
        # make weights torch parameters
        self.weights_L = torch.nn.Parameter(weights_L)
        self.weights_R = torch.nn.Parameter(weights_R)
        self.eps = eps
        self.delta = delta

    def forward(self):
        q_hat = torch.matmul(self.weights_L, torch.transpose(self.weights_R, 0, 1))
        return q_hat, self.weights_L, self.weights_R

    def pca_initialization(self, query_matrix, k_components):
        svd = np.linalg.svd(query_matrix, full_matrices=False, compute_uv=True)
        U = svd[0]
        s = svd[1]
        Vh = svd[2]
        U_k = U[:, range(k_components)]
        S_k = np.diag(s[range(k_components)])
        V_kh = torch.tensor(Vh[range(k_components), :])
        self.weights_L = torch.nn.Parameter(torch.tensor(np.matmul(U_k, S_k)))
        self.weights_R = torch.nn.Parameter(torch.transpose(V_kh, 0, 1))


# If M = LR^T-X, and if our loss fn is the MSE then the loss can be upper bounded by
# 1/m * (||M||_F^2*1/n + 1/n^2||L||_F^2||R||_{1->2}^2*privacy_blah
# Frobenius norms can be replaced with the spectral norm, set p = '2'
def loss_fn_l_2(L, R, q_ids, epsilon, delta, ids=None, silent=True, p='fro'):
    m = L.shape[0]
    n_samp = q_ids.shape[0]
    if ids is None:
        approximation_loss = 1/m*(1/n_samp*torch.pow(torch.linalg.matrix_norm(torch.matmul(L, torch.transpose(R, 0, 1)) - q_ids, ord=p), 2))
    else:
        approximation_loss = 1/m*(1/n_samp*torch.pow(torch.linalg.matrix_norm(torch.matmul(L[ids,:], torch.transpose(R, 0, 1)) - q_ids, ord=p), 2))

    sigma = gaussian_mech_noise(torch.max(torch.norm(R, dim=1)), epsilon, delta, tensor=True)
    noise_loss = (1/np.power(n, 2)) * torch.pow(torch.linalg.matrix_norm(L, ord=p), 2) * torch.pow(sigma, 2)

    if not silent:
        print('mse: {}, mse of appx: {}, per coord noise: {}'.format(approximation_loss.detach() +
                                                                                     noise_loss.detach(),
                                                                                     approximation_loss.detach(),
                                                                                     noise_loss.detach()))
    pdb.set_trace()
    return approximation_loss + noise_loss


def training_loop(dataloader, model, loss, optimizer):
    "Training loop for torch model."
    for batch, (q, ids) in enumerate(dataloader):
        # Compute prediction and loss
        qhat, L, R = mm_fac()
        loss = loss_fn_l_2(L, R, q, model.eps, model.delta, ids)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_fn_l_2(L, R, training_data.query_matrix, model.eps, model.delta, ids=None, silent=False)



if __name__ == "__main__":
    training_data = SNPDataset(query_matrix)
    bsize = 128
    epochs = 1000
    train_dataloader = DataLoader(training_data, batch_size=bsize, shuffle=False)
    m, n = query_matrix.shape
    k = min(m, n)
    # (eps, delta) effect the loss fn, so relevant even though this is not private opt
    eps = 10.0
    delta = 1.0/np.power(n,2)
    mm_fac = Factorization(m, n, k, eps=eps, delta=delta)
    mm_fac.pca_initialization(query_matrix, k_components=k)
    learning_rate = 1.0
    optimizer = torch.optim.SGD(mm_fac.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(train_dataloader, mm_fac, loss_fn_l_2, optimizer)
    print("Done!")

# LEFT OFF FEELS LIKE NN is not optimizing the right things