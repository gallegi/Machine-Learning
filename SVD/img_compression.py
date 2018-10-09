import numpy as np

import cv2

image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imwrite('2.jpg', cv2.resize(image, (1000, 750)))

#svd
U, S, V = np.linalg.svd(image)

n = 1000
m = 750

k = 200

S_matrix = np.zeros((m,n))
S_matrix[np.arange(m), np.arange(m)] = S

img = (U.dot(S_matrix).dot(V)).astype(np.uint8)

cv2.imshow('image', image)
cv2.imshow('img', img)

print(np.linalg.norm(img - image, 'fro'))


# processing to compress image k = 100

S_comp = np.zeros((k,k))
S_comp[np.arange(k), np.arange(k)] = S[:k]
img_comp = (U[:,:k].dot(S_comp).dot(V[:k,:])).astype(np.uint8)

cv2.imshow('comp', img_comp)

cv2.waitKey(0)