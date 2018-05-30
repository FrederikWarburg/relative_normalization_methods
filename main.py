#############
# Import packages
#############

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from relative_normalization_methods import *

#############
# Generate Synthetic Data
#############

mean = (50, 75)
sigma = [[500, 50], [250, 1000]]

x = multivariate_normal(mean, sigma)

m,n = 99,150

im = np.ones((m,n))

for i in range(m):
    for j in range(n):
        im[i,j] = x.pdf((i,j))*1000000

#plt.imshow(im,vmin=0,vmax=255)
#plt.axis('off')
#plt.title('Final image')
#plt.show()

# slice image into 9
slices = 9

im_slices = []

m,n = 100,150

for i in range(3):
    for j in range(3):
        from_h = int(np.floor(m/3*i))
        to_h = int(np.floor((m)/3*(i+1)))
        from_v = int(np.floor(n/3*j))
        to_v = int(np.floor((n)/3*(j+1)))
        tmp = im[from_h:to_h,from_v:to_v]
        im_slices.append(tmp)

#plt.figure(figsize=(10,7.5))
#for i in range(slices):
#    plt.subplot(3,3,i+1)
#    plt.imshow(im_slices[i],vmin=0,vmax=255)
#    plt.axis('off')
#    plt.title('Image {}'.format(i))
#plt.show()

# adjust image intensities such that we later can correct (normalize the intensities)
factors = [5,0.2,1.3,
           1.8,1,0.1,
           5.1,0.2,10]

for i in range(9):
    im_slices[i] = factors[i]*im_slices[i]

plt.figure(figsize=(10,7.5))
for i in range(slices):
    plt.subplot(3,3,i+1)
    plt.imshow(im_slices[i],vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Image {}'.format(i))
plt.show()


##########
# Iteration 1
# Use relative normalization methods to compensate for the adjustments in intensities for the bordering images to
# the center image.
##########

# copy image
im_slices_copy = im_slices

# image 4 master im
master_im = im_slices[4]
slave_im = im_slices[1]
position = 0

M, factor = relative_normalization_one_image(master_im,slave_im, position)

im_slices_copy[1] = im_slices[1]*M

slave_im = im_slices[7]
position = 1
M, factor = relative_normalization_one_image(master_im,slave_im, position)
im_slices_copy[7] = im_slices[7]*M

slave_im = im_slices[5]
position = 2
M, factor = relative_normalization_one_image(master_im,slave_im, position)
im_slices_copy[5] = im_slices[5]*M

slave_im = im_slices[3]
position = 3
M, factor = relative_normalization_one_image(master_im,slave_im, position)
im_slices_copy[3] = im_slices[3]*M

plt.figure(figsize=(10,7.5))
for i in range(slices):
    plt.subplot(3,3,i+1)
    plt.imshow(im_slices_copy[i],vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Image {}'.format(i))
plt.show()

##########
# Iteration 2
# Use relative normalization methods to compensate for the adjustments in intensities for the two bordering images to
# the remainding images.
##########

# Positions:
# 1 2 3
# 4 5 6
# 7 8 9
#
# 0: master_im1 = 2, master_im2 = 6, slave_im = 3
# 1: master_im1 = 6, master_im2 = 8, slave_im = 9
# 2: master_im1 = 8, master_im2 = 4, slave_im = 7
# 3: master_im1 = 4, master_im2 = 2, slave_im = 1

# position 1
master_im1 = im_slices_copy[5]
master_im2 = im_slices_copy[7]
slave_im = im_slices[8]
position = 1

M = relative_normalization_two_images(master_im1, master_im2, slave_im, position)
im_slices_copy[8] = im_slices[8]*M

# position 3
master_im1 = im_slices_copy[3]
master_im2 = im_slices_copy[1]
slave_im = im_slices[0]
position = 3

M = relative_normalization_two_images(master_im1, master_im2, slave_im, position)
im_slices_copy[0] = im_slices[0]*M

# position 2
master_im1 = im_slices_copy[7]
master_im2 = im_slices_copy[3]
slave_im = im_slices[6]
position = 2

M = relative_normalization_two_images(master_im1, master_im2, slave_im, position)
im_slices_copy[6] = im_slices[6]*M

# position 0
master_im1 = im_slices_copy[1]
master_im2 = im_slices_copy[5]
slave_im = im_slices[2]
position = 0

M = relative_normalization_two_images(master_im1, master_im2, slave_im, position)
im_slices_copy[2] = im_slices[2]*M

plt.figure(figsize=(10,7.5))
for i in range(slices):
    plt.subplot(3,3,i+1)
    plt.imshow(im_slices_copy[i],vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Image {}'.format(i))
plt.show()
