from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

# Parameters
num_bins = 40
tile_size = 80
stddev = 0.017359

# Load the existing umap projection
projection = np.load('umap_projection.npy')

# Load the dataset and save as list (so it can be indexed)
def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

initial_dataset = tfds.load('hsc_icl', split='train')
dataset = initial_dataset.map(preprocess)
ds_list = list(dataset.as_numpy_iterator())

# Bin the data
hist, xedges, yedges, _ = plt.hist2d(
    projection[:,0], projection[:,1], bins=num_bins
)
plt.close()

# Get an example from every non-zero bin and save the coordinates of that bin
x, y = np.where(hist > 0)
coords = list(zip(x, y)) # Coords of non-zero elements

images = []
for c in coords:
    # Get the lower left and upper right corners of the bin
    ll = np.array([xedges[c[0]], yedges[c[1]]])
    ur = np.array([xedges[c[0]+1], yedges[c[1]+1]])

    # Get an example cluster that falls inside this box
    idx = np.nonzero(np.all(
        np.logical_and(ll <= projection, projection <= ur),
        axis=1
    ))[0][0]

    images.append(ds_list[idx])

size = tile_size * num_bins
output = Image.new('RGB', (size, size), (255,255,255)) # White background

# Draw tiles
for i, loc in enumerate(coords):
    # Convert to PIL image and resize
    img_norm = (images[i] / np.max(images[i])).squeeze()
    coloured = np.uint8(cm.viridis(img_norm) * 255) # Using viridis colourmap
    img = Image.fromarray(coloured).convert('RGB')
    img = img.resize((tile_size, tile_size))

    # Calculate location of this tile and paste
    x, y = loc[0] * tile_size, loc[1] * tile_size 
    output.paste(img, (x,y))

# Convert back into array so we can imshow, also save the output png
output_arr = np.asarray(output)

output = output.transpose(Image.FLIP_TOP_BOTTOM) # flip to match histogram shape
output.save('umap_tiled.png')

# Put rectangular patches over each cluster. Indices of rects correspond to 
# coords and images
rects = []
offset = 20
for c in coords:
    # xy is lower left
    rect = Rectangle((c[0] * tile_size + offset // 2, 
                      c[1] * tile_size + offset // 2), 
                      tile_size - offset, tile_size - offset)
    rects.append(rect)
    
pc = PatchCollection(rects, color='none')

# Create figure and plot patches
fig = plt.figure(figsize=(8,8), dpi=100)
ax = fig.add_subplot(111)
colours = [np.random.uniform(size=(3)) for _ in range(len(rects))]
ax.imshow(output_arr, origin='lower')
ax.add_collection(pc)

arr = np.random.rand(len(images), 10, 10)

# Create the annotations box
im = OffsetImage(images[0], zoom=0.7, origin='lower')
xybox=(85., 85.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
                    boxcoords='offset points', pad=0.3, animated=True)
# Add to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

plt.show(block=False)
plt.pause(0.5) # Need to pause to wait for background to be fully drawn

# Need to use blitting to avoid redrawing the entire background image every time
bg = fig.canvas.copy_from_bbox(fig.bbox)
fig.canvas.blit(fig.bbox)

def hover(event):
    fig.canvas.restore_region(bg)
    # If the mouse is over the scatter points
    if pc.contains(event)[0]:
        # Find the index within the array from the event
        ind = pc.contains(event)[1]['ind'][0]
        # Get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # If event occurs in the top or right quadrant of the figure, change the 
        # annotation box position relative to mouse
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # Make annotation box visible
        ab.set_visible(True)
        # Place it at the position of the hovered scatter point
        ab.xy = (coords[ind][0]*tile_size, coords[ind][1]*tile_size)
        # Set the image corresponding to that point
        im.set_data(images[ind])
    else:
        # If the mouse is not over a scatter point
        ab.set_visible(False)

    ax.draw_artist(ab)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

# Add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)
plt.show()