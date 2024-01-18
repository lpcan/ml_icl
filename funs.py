import matplotlib.pyplot as plt
import numpy as np
import pickle

MODEL_VERSION = 'nobatchnorm-final'

def flatten(list):
    return np.array([i for row in list for i in row])

# Get fracs into the same order as results
fracs = np.load('/srv/scratch/mltidal/fracs_resized.npy')[2]
zeros = np.where(fracs == 0)[0]
if len(zeros) > 0:
    fracs[zeros] = 0.00001

rng = np.random.default_rng(seed=24)
idxs = np.arange(len(fracs))
rng.shuffle(idxs)

fracs = fracs[idxs]

# Bin the data
with open(f'/srv/scratch/mltidal/finetuning_results/{MODEL_VERSION}.pkl', 'rb') as fp:
    results, err_l, err_h = pickle.load(fp)

results = flatten(results)
err_l = flatten(err_l)
err_h = flatten(err_h)
sorted_idxs = np.argsort(fracs)

binned_results = np.array_split(results[sorted_idxs], 5)
binned_fracs = np.array_split(fracs[sorted_idxs], 5)

# Calculate the median of the binned results
x = []
y = []
xerr_l = []
xerr_h = []
for i in range(len(binned_results)):
    x_med = np.median(binned_fracs[i])
    x.append(x_med)
    y_med = np.median(binned_results[i])
    y.append(y_med)

    xerr_l.append(x_med - np.min(binned_fracs[i]))
    xerr_h.append(np.max(binned_fracs[i]) - x_med)

plt.plot(fracs, results, '.', color='gray', alpha=0.3)
plt.errorbar(fracs, results, fmt='none', yerr=(err_l, err_h), alpha=0.2, color='gray')
plt.plot(x, y, 'or')
plt.plot(x, y, 'r')
plt.errorbar(x, y, fmt='none', xerr=(xerr_l, xerr_h), color='red', alpha=0.3)
plt.xlabel('Expected')
plt.ylabel('Predicted')

maxval = np.max([fracs, results])
plt.plot([0,maxval], [0,maxval], 'k--')

plt.savefig('asdf.png')
