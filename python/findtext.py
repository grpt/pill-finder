# coding: utf-8

import numpy as np
import cv2


def findtext(imgcolorc, imgcolorf):
    img = imgcolorc
    img2 = imgcolorf
    row, col, _ = img.shape
    row2, col2, _ = img2.shape
    distrow = (int)(row * 0.2)
    distcol = (int)(col * 0.2)
    distrow2 = (int)(row2 * 0.2)
    distcol2 = (int)(col2 * 0.2)
    img = img[0 + distrow:row - distrow, 0 + distcol:col - distcol]
    img2 = img2[0 + distrow2:row2 - distrow2, 0 + distcol2:col2 - distcol2]

    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col, _ = img.shape
        # time.sleep(1)
    grayimage2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(grayimage, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(grayimage, cv2.CV_64F, 0, 1)

    sobelX2 = cv2.Sobel(grayimage2, cv2.CV_64F, 1, 0)
    sobelY2 = cv2.Sobel(grayimage2, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelX2 = np.uint8(np.absolute(sobelX2))
    sobelY2 = np.uint8(np.absolute(sobelY2))

    grayimage = sobelX + sobelY
    grayimage2 = sobelX2 + sobelY2

    row, col = grayimage.shape
    for i in range(0, row):
        for j in range(0,col) :
            if grayimage2[i][j] == 0 and grayimage[i][j] !=0:
                grayimage[i][j] == 0
    for i in range(0, row):
        for j in range(0, col):
            if grayimage[i][j] >= grayimage[i][j] - grayimage2[i][j]:
                grayimage[i][j] = grayimage[i][j] - grayimage2[i][j]
    sum = 0
    for i in range(0, row):
        for j in range(0, col):
            sum = sum + grayimage[i][j]

    sum = (int)(sum / (row * col))
    sum = sum + (int)(sum * 0.1)

    grayimage = cv2.GaussianBlur(grayimage, (3, 3), 0)
    _, th = cv2.threshold(grayimage, sum, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max = 0
    for i in range(1, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= max:
            max = w * h
    textarr = []
    origarr = []
    xarr = []
    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < (int)(max * 0.3):
            continue
        tmp = th[y:y + h, x:x + w]
        textarr.append(tmp.copy())
        tmp = img[y:y + h, x:x + w]
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        origarr.append(tmp.copy())
        xarr.append(x)

    xlength = len(xarr)
    for i in range(0, xlength):
        for j in range(0, xlength):
            if xarr[i] < xarr[j]:
                tmp = xarr[i]
                xarr[i] = xarr[j]
                xarr[j] = tmp

                tmp = origarr[i].copy()
                origarr[i] = origarr[j].copy()
                origarr[j] = tmp.copy()

                tmp = textarr[i].copy()
                textarr[i] = textarr[j].copy()
                textarr[j] = tmp.copy()
    return origarr, textarr