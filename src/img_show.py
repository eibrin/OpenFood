import cv2

img_file = '../res/pose01.jpg'
img_save_file = '../res/pose01_greyscale.jpg'

img = cv2.imread(img_file)
img_grey = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)


if img is not None:
    cv2.imshow('IMG', img)
    cv2.waitKey(0)
    cv2.imwrite(img_save_file, img)
    cv2.destroyAllWindows()
else:
    print('Image not found')
