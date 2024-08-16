"""
Create a table of all successfully downloaded cutouts from both DUD and WIDE
with consistent IDs
"""

from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import vstack
import numpy as np

dud_path = "../raw/camira_s20a_dud.tbl"
wide_path = "../raw/camira_s20a_wide.tbl"
failed_path = "../raw/cutouts/failed_downloads.txt"

# Start with the DUD clusters
table = ascii.read(dud_path,  names = ["ID", "Name", "RA [deg]", "Dec [deg]", "z_cl", "Richness", "BCG redshift"])
table = table[table["z_cl"] <= 0.5] # Filter
print(f"{len(table)} DUD clusters")

# Append the Wide clusters
table_wide = ascii.read(wide_path, names = ["ID", "Name", "RA [deg]", "Dec [deg]", "z_cl", "Richness", "BCG redshift"])
table_wide = table_wide[table_wide["z_cl"] <= 0.5]
print(f"{len(table_wide)} wide clusters")

# Cross match and remove matches from wide catalogue
dud_coords = SkyCoord(table["RA [deg]"], table["Dec [deg]"], unit=u.deg)
wide_coords = SkyCoord(table_wide["RA [deg]"], table_wide["Dec [deg]"], unit=u.deg)
_, idx, _, _ = wide_coords.search_around_sky(dud_coords, 25*u.arcsec)
mask = np.ones(len(wide_coords), dtype=bool)
mask[idx] = 0
table_wide = table_wide[mask]
print(f"{len(table_wide)} wide clusters after cross-matching")

# Create two lists of the failed downloads
f = open(failed_path)
failed = [[], []]
cat = -1

for line in f:
    words = line.split(' ')
    if words[0] == "Clusters":
        cat += 1
    else:
        idx = line.split(' ')[1]
        failed[cat].append(int(idx))

# Mask out the failed downloads in both tables
mask = np.ones(len(table), dtype=bool)
mask[failed[0]] = 0
table = table[mask]

mask = np.ones(len(table_wide), dtype=bool)
mask[failed[1]] = 0
table_wide = table_wide[mask]

# Concatenate the two catalogues
table = vstack([table, table_wide])
print(f"{len(table)} successful cutouts")

# Finally, mask out the bad images (identified by visual inspection) from the combined catalogue
bad = [32, 56, 114, 126, 130, 167, 254, 428, 535, 545, 584, 592, 623, 625, 653, 654, 657, 666, 696, 851, 874, 877, 879, 884, 889, 895, 903, 913, 926, 938, 949, 954, 970, 972, 973, 977, 1000, 1060, 1069, 1096, 1104, 1105, 1106, 1108, 1110, 1118, 1119, 1132, 1133, 1134, 1136, 1137, 1138, 1139, 1145, 1146, 1151, 1160, 1167, 1173, 1199, 1207, 1214, 1217, 1252, 1259, 1263, 1266, 1275, 1284, 1286, 1290, 1303, 1326, 1334, 1342, 1358, 1383, 1384, 1395, 1399, 1402, 1406, 1407, 1549, 1672, 1678, 1788, 1794, 1817, 1819, 1830, 1834, 1846, 1890, 2071, 2305, 2409]
mask = np.ones(len(table), dtype=bool)
mask[bad] = 0
table = table[mask]

print(f"{len(table)} good cutouts")

# Re-ID all the clusters
table["ID"] = list(range(len(table)))

# Update the table header
table.meta["comments"] = ["Final cluster catalogue of successfully downloaded clusters", "column 1: ID", 
                          "column 2: Name", "column 3: RA [deg]", "column 4: Dec [deg]", 
                          "column 5: cluster photometric redshift z_cl", "column 6: richness N_mem", 
                          "column 7: spectroscopic redshift of BCG (-1.0 if not available)"]

table.write("camira_final.tbl", format="ascii", overwrite=True)


