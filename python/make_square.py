import cv2

def make_square(img):
    row,col = img.shape
    if row > col:
        size = row
    else:
        size = col

    rowBorder = (int)((size - row)/2)
    colBorder = (int)((size - col)/2)

    dest = cv2.copyMakeBorder(img, rowBorder, rowBorder, colBorder, colBorder, cv2.BORDER_CONSTANT, value = [0,0,0])

    return dest

