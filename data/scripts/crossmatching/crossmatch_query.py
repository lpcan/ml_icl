"""
Check how many coordinates in the LRG wide member table are within the DUD 
footprint of the HSC survey and save as a table.
"""

from astropy.io import fits
from astropy.table import Table, vstack

from hscSspCrossMatch import main
from hscReleaseQuery import sql_query

LRG_CATALOGUE_PATH = 'TODO:LRG_CATALOGUE_PATH (FROM https://drive.google.com/drive/folders/19dKiNs7Wdq44X3AtYy2opOf8ulHTLWeM)'

f = fits.open(LRG_CATALOGUE_PATH)
tbl = f[1].data[f[1].data['z'] <= 0.5]
tbl = Table(tbl)

start = 0
end = 10000

results = Table(names=['object_id', 'user_ra', 'user_dec', 'user_ms', 'user_z'])

# Set environment variable 'HSC_SSP_CAS_PASSWORD' to HSC password to avoid having to keep reentering it

while start < len(tbl):
    query = main(tbl[start:end], 'pdr2_dud') # Generates crossmatch query for this table in r-band
    result = sql_query(query, user='locan@local', release_version='pdr2') # Submit the query
    results = vstack([results, result])
    end += 10000
    start += 10000

results.write('lrg_dud_crossmatch.tbl', format='ascii', overwrite=True)
