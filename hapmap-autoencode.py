import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import pickle

# please set the path to your data directory here
data_path = '/Users/sneel/Dropbox/Research/Current_Projects/epi-data/Data'
from gen_helpers import *
# read in genotype data from CEU population (112 respondents, 718848 snps)
geno_matrix = read_geno(pname('CEU.geno', data_path))

# remove SNPs with missing data: 500k by 112
geno_matrix = np.ma.compress_rows(geno_matrix)

# goal privately compute the allele frequency:
# 1. Learn low dimensional representation (Auto-Encoder?)
# 2. Compute average loadings
# 3. Convert back to average in high dimensional space
# 4. Assess accuracy

# compute principle components
# transpose
geno_snps_cols = np.transpose(geno_matrix)
pca = PCA()
pca.fit(geno_snps_cols)

# note: inspecting the explained variance, this is not approximately lower rank than 112, top PC only explains 1.3% of variance

# downsample 1000 columns. Note that d = 10*n is a common ratio eg 1000 genomes n = 3k, p >= 30k in the HLA region
# We use the autoencoder of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6769581/
slice_geno = geno_snps_cols[:, 0:1000]
slice_geno = torch.tensor(slice_geno, dtype=torch.float64)
test_geno = slice_geno[1:10,:]
train_geno = slice_geno[10:,:]


# create the torch dataset
class GenomicDataSet(Dataset):
    def __init__(self, geno_matrix):
        self.data = geno_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :].float()

training_geno = GenomicDataSet(train_geno)
train_data_loader = DataLoader(training_geno, batch_size=8, shuffle=True)
test_data_loader = DataLoader(test_geno, batch_size=8, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define Encoder and Decoder Models
class Encoder(nn.Module):
    def __init__(self, dim_hidden=10):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 1000)
            #nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLU(),
           # nn.Linear(512, dim_hidden)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class Decoder(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            #nn.Linear(dim_hidden, 512),
            #nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Linear(1000, 1000)
        )

    def forward(self, x):
        return self.decoder_lin(x)


# Define Loss Function and Optimizer
loss_fn = torch.nn.MSELoss()
# set learning rate
lr = .01
# set for reproducbility
torch.manual_seed(0)

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
hidden_size = 1000
encoder = Encoder(dim_hidden=hidden_size)
decoder = Decoder(dim_hidden=hidden_size)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for genome_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        genome_batch = genome_batch.to(device)
        # Encode data
        encoded_data = encoder(genome_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, genome_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

num_epochs = 1000
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss = train_epoch(encoder,decoder,device,
   train_data_loader,loss_fn,optim)
   #val_loss = test_epoch(encoder,decoder,device,test_data_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
   diz_loss['train_loss'].append(train_loss)
   #diz_loss['val_loss'].append(val_loss)

for name, param in encoder.named_parameters():
    if param.requires_grad:
        print(name, param.data)


