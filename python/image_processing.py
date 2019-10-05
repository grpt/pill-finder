# -*- coding: utf-8 -*-

import cv2
import numpy as np


def backproject(source, target, levels=2, scale=1):
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv2.calcHist([hsv], [0, 1], None, [levels, levels], [0, 180, 0, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], scale)
    return dst


def text(img):
    imgx = img

    imglast = imgx.copy()

    row = 0
    col = 0
    row, col, _ = imgx.shape

    cntb = 0
    cntf = 0

    #   kernel size 3으로 설정

    kernel = np.ones((3, 3), np.uint8)
    imgx1 = cv2.morphologyEx(imgx, cv2.MORPH_CLOSE, kernel)

    #   kernel size 증가하면서 전체 크기 변할시 그 전 값 리턴

    for ksize in range(3, row, 2):
        kernel = np.ones((ksize, ksize), np.uint8)
        imgx2 = cv2.morphologyEx(imgx, cv2.MORPH_CLOSE, kernel)

        b, g, r = cv2.split(imgx2)

        for i in range(int(row / 2)):
            for j in range(int(col / 2)):
                if b[i][j] + g[i][j] + r[i][j] > 30:
                    cntf += 1
        if cntb is 0:
            cntb = cntf
        elif cntf - cntb > int(cntb * 0.02):
            break
        imgx1 = imgx2
        cntf = 0

    return imgx1


def fill(imgx, imglast, threshMap, img_size):
    row, col = img_size
    kernel = np.ones((3, 3), np.uint8)
    threshMap1 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)

    for i1 in range(row):
        if threshMap1[i1][(int)(col / 2)] > 20:
            break
    for j1 in range(col):
        if threshMap1[(int)(row / 2)][j1] > 20:
            break
    size = 5
    while (size < 20):

        kernel = np.ones((size, size), np.uint8)
        threshMap2 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)

        size += 2
        for i2 in range(row):
            if threshMap2[i2][(int)(col / 2)] > 20:
                break
        for j2 in range(col):
            if threshMap2[(int)(row / 2)][j2] > 20:
                break
        if i1 != i2 or j1 != j2:
            break
        else:
            threshMap1 = threshMap2
            i1 = i2
            j1 = j2

    threshMap = threshMap1
    threshMapfilter = cv2.GaussianBlur(threshMap, (7, 7), 0)
    imglast2 = imglast.copy()
    for i in range(row):
        for j in range(col):
            if (threshMap[i][j] == 0):
                imgx[i][j] = 0
                imglast[i][j] = 0
    for i in range(row):
        for j in range(col):
            if (threshMapfilter[i][j] == 0):
                imglast2[i][j] = 0
    return imglast, imglast2, threshMap, threshMapfilter


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if abs(mx - mn) < 0.04:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v * 100


def color(bbb):
    b, g, r = cv2.split(bbb)
    cb = 0
    cg = 0
    cr = 0
    cnt = 0
    h, w, _ = bbb.shape
    for i in range(h):
        for j in range(w):
            if (b[i][j] != 0) and (g[i][j] != 0) and (r[i][j] != 0):
                cb += b[i][j]
                cg += g[i][j]
                cr += r[i][j]
                cnt += 1
    cb = (int)(cb / cnt)
    cg = (int)(cg / cnt)
    cr = (int)(cr / cnt)
    #     for i in range(h):
    #         for j in range(w):
    #             if (b[i][j] < 20):
    #                 b[i][j] = cb
    #             if (g[i][j] < 20):
    #                 g[i][j] = cg
    #             if (r[i][j] < 20):
    #                 r[i][j] = cr

    h, _, v = rgb2hsv(cr, cg, cb)
    # 검은색, 회색, 흰색 해야됨
    if abs(cb - cg) + abs(cg - cr) + abs(cr - cb) < 50:
        if v < 30:
            color1 = 'black'
        elif v < 70:
            color1 = 'gray'
        else:
            color1 = 'white'
    else:
        if h <= 60 or h > 330:
            color1 = 'red'
        elif h <= 180:
            color1 = 'green'
        elif h <= 225:
            color1 = 'blue'
        elif h <= 330:
            color1 = 'purple'

    bbb = cv2.merge([b, g, r])
    return bbb, color1


# 약 설정
def image_processing(img):
    # 약 설정
    # imgx = img[0:370, :]
    imgx = img
    imglast = imgx.copy()
    cv2.pyrMeanShiftFiltering(imgx, 2, 10, imgx, 4)

    backproj = np.uint8(backproject(imgx, imgx, levels=2))
    backproj = 255 - backproj


    cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)

    saliencies = [backproj, backproj, backproj]
    saliency = cv2.merge(saliencies)

    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)

    saliencyMap = saliency

    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # hsv로 변환후 v가 0.5이하면 그림자로 취급 검은색으로 만듬
    hsv = cv2.cvtColor(imgx, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    (row, col) = v.shape

    # 최대 v 최소v 찾고 0에서 1로 정규화
    max = v[0][0]
    min = v[0][0]

    ave = 0
    cnt = 0

    for i in range(row):
        for j in range(col):
            ave += v[i][j]
            cnt += 1
            if (max < v[i][j]):
                max = v[i][j]
            elif (min > v[i][j]):
                min = v[i][j]

    # 그림자 제거
    (row, col) = threshMap.shape
    for i in range(row):
        for j in range(col):
            x = (v[i][j] - min) / ((max - min) + 1e-10)
            if (x <= 0.3):
                threshMap[i][j] = 0
    # fill로 채우고 경계부분 검출

    imglast, imglast2, threshMap, threshMapfilter = fill(imgx, imglast, threshMap, (row, col))



    # contours, hierarchy = cv2.findContours(threshMap * 1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # contours = sorted(contours, key=cv2.contourArea)

   # x, y, w, h = cv2.boundingRect(contours[-1])

   # imgcolorc = imglast[y:y + h, x:x + w]

    contours, hierarchy = cv2.findContours(threshMapfilter * 1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea)

    # ------------------------------크기 정사각형 만드는 부분------------------------------------------
    flag = 0
    size = len(contours) -1
    for j in range(0,1000):
        for i in range(-1, size):
            x, y, w, h = cv2.boundingRect(contours[i])
            if (x - j*10< 210 and x + w  + j*10> 210 and y - j*10< 280 and y + h + j*10> 280):
                flag =1
                break
        if flag == 1:
            break
    imgcolorc = imglast[y:y + h, x:x + w]
    if (h > w):
        x = x - (int)((h - w) / 2)
        w = h
    else:
        y = y - (int)((w - h) / 2)
        h = w
    # ------------------------------------------------------------------------------

    imgshape = threshMapfilter[y:y + h, x:x + w]

    imgcolorc, colorname = color(imgcolorc)
    imgcolorf = text(imgcolorc)

    return imgcolorc, imgcolorf, imgshape, colorname