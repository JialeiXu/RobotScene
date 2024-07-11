import cv2
import numpy as np
img = cv2.imread('img.png')
print(img.shape)
h,w,_ = img.shape
for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        a,b,c = img[h][w]
        if a>230 and b>230 and c>230:
            img[h][w] = 255,255,255
cv2.imwrite('img.jpg',img)