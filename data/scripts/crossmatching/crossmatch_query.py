from astropy.io import fits
from astropy.table import Table, vstack

from hscSspCrossMatch import main
from hscReleaseQuery import sql_query

f = fits.open('/srv/scratch/z5214005/lrg_s18a_wide_sm.fits')
tbl = f[1].data[f[1].data['z'] <= 0.5]
tbl = Table(tbl)

start = 0
end = 10000

results = Table(names=['object_id', 'user_ra', 'user_dec', 'user_ms', 'user_z'])

while start < len(tbl):
    query = main(tbl[start:end], 'pdr2_dud')
    result = sql_query(query, user='locan@local', release_version='pdr2')
    results = vstack([results, result])
    end += 10000
    start += 10000

results.write('lrg_dud_crossmatch.tbl', format='ascii', overwrite=True)
