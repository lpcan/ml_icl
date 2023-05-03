"""
Create a single HDF file with all the downloaded cutouts that can be used to create a TensorFlow dataset
"""

from fits2hdf.io.hdfio import export_hdf
from fits2hdf.io.fitsio import read_fits
import glob
import h5py
import os
import re
import shutil

script_dir = os.path.dirname(__file__) # Want paths relative to the script, not relative to current working directory
cutouts_path = os.path.join(script_dir, '../../cutouts_mask_var/*.fits')#"../raw/cutouts/*.fits") # Where the individual downloaded cutouts are located
output_path = os.path.join(script_dir, '../../') #"../processed/") # Where to put the final HDF5 file

# Functions to enable sorting in "natural order" (i.e. same order as is in the catalogue). Code from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# Create a temporary directory to store the converted files - note this will fail if tmp/ directory already exists!
tmp_dir_path = os.path.join(script_dir, "tmp")
os.mkdir(tmp_dir_path)

# Transform each file into HDFFITS format and save into temp directory - here we also assign them IDs consistent with final catalogue
fnames = sorted(glob.glob(cutouts_path), key=natural_keys) # Sort in natural order to put in the same order as catalogue - DUD first, sorted by RA
for i,name in enumerate(fnames):
    f = read_fits(name)
    export_hdf(f, f"{tmp_dir_path}/{i}.hdf")

# Aggregate all files 
with h5py.File(output_path+"cutouts.hdf", mode='w') as d:
    for i in range(len(fnames)):
        with h5py.File(f"{tmp_dir_path}/{i}.hdf", mode='r+') as s:
            d.copy(s, f"{i}")

# Remove the temporary directory
shutil.rmtree(tmp_dir_path)