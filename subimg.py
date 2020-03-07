import cv2
import matplotlib.pyplot as plt

subwidth = 36

img = cv2.imread('img.jpg')
srcH = img.shape[0]
srcW = img.shape[1]

sH = int((0.1738 * srcH/srcW+0.1736)*srcH)

img = img[sH:sH + srcW // 6 * 5, :, :]
print(img.shape)
img = cv2.resize(img, (subwidth * 6, subwidth * 5))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for i in range(5):
    for j in range(6):
        sh = subwidth * i
        subimg = img[sh:sh + subwidth, j * subwidth:(j + 1) * subwidth, :]
        [b, g, r] = cv2.split(subimg)
        plt.imshow(cv2.merge([r, g, b]))
        plt.show()
        name = input()
        if name == '-':
            continue
        filename = name + '.png'
        cv2.imwrite("subpic/" + filename, subimg)
