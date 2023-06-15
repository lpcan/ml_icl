import torch
import clip
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import skimage
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm

# Custom transform
class ArcsinhScale(object):
    def __call__(self, image):
        image = np.clip(image, a_min=0, a_max=10)
        image = np.arcsinh(image / 0.017359)
        image = image / np.max(image) * 255
        return image

# Custom dataset class
class ClusterDataset(data.Dataset):
    def __init__(self, cutouts_file, transform=None):
        self.cutouts = h5py.File(cutouts_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.cutouts.keys())
    
    def __getitem__(self, idx):
        cutout = np.array(self.cutouts[str(idx)]['HDU0']['DATA'], dtype=np.float32)
        cutout = np.expand_dims(cutout, axis=2) # add channel axis
        
        if self.transform:
            cutout = self.transform(cutout)

        return cutout

# Compute similarity between embeddings `p` (2D) and embedding `z` (1D)
def compute_similarity(p, z):
    p = p / np.expand_dims(np.linalg.norm(p, axis=1), axis=-1) # replicate tf.math.l2_normalize
    z = z / np.linalg.norm(z)

    return np.sum(p * z, axis=1)

# Function to calculate Spearman coefficient
fracs_path = '/srv/scratch/z5214005/fracs.npy'
def calc_spearman(embeddings):
    # Rank the calculated fractions
    fracs = np.load(fracs_path)
    fracs_ordered = np.argsort(fracs[2])
    mask = ~np.isnan(fracs[2][fracs_ordered])
    fracs_ordered = fracs_ordered[mask][::-1]
    fracs_ranked = np.argsort(fracs_ordered)

    # Rank model embeddings with respect to similarity to cluster 51
    cluster_51 = np.expand_dims(embeddings[51], 0)
    similarities = compute_similarity(embeddings, cluster_51)
    ordered = np.argsort(similarities)[::-1]
    ordered = ordered[~np.isnan(fracs[2][ordered])]
    rankings = np.argsort(ordered)

    return spearmanr(rankings, fracs_ranked).statistic

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device=device)

# Add the arcsinh stretch to the preprocessing transforms
preprocess.transforms.insert(0, ArcsinhScale())
preprocess.transforms.insert(1, transforms.ToPILImage())

# Create the dataset
dataset = ClusterDataset(cutouts_file='/srv/scratch/z5214005/hsc_icl/cutouts.hdf',
                         transform=preprocess)

# Create dataset loader 
all_features = []

with torch.no_grad():
    for images in tqdm(data.DataLoader(dataset, batch_size=100)):
        features = model.encode_image(images.to(device))
        all_features.append(features)

features = torch.cat(all_features).cpu().numpy()

print(features.shape)

dud_features = features[:125]

print(calc_spearman(dud_features))