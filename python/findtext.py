# coding: utf-8

import numpy as np
import cv2


def findtext(imgcolorc, imgcolorf):
    #imgcolorc : 원본이미지를 도형에 맞게 컷팅해준 사진이다.
    #imgcolorf : imgcolorc사진의 글자 부분을 지운 사진이다.
    
    img = imgcolorc
    img2 = imgcolorf
    row, col, _ = img.shape
    row2, col2, _ = img2.shape
    distrow = (int)(row * 0.2)
    distcol = (int)(col * 0.2)
    distrow2 = (int)(row2 * 0.2)
    distcol2 = (int)(col2 * 0.2)
    img = img[0 + distrow:row - distrow, 0 + distcol:col - distcol] #이미지의 크기를 설정
    img2 = img2[0 + distrow2:row2 - distrow2, 0 + distcol2:col2 - distcol2] #이미지의 크기를 설정

    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #이미지를 GrayScale(0~255 까지 밝기의 농도를 가진 사진)로 변경한다.
    row, col, _ = img.shape
    
    grayimage2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #이미지를 GrayScale(0~255 까지 밝기의 농도를 가진 사진)로 변경한다.

    sobelX = cv2.Sobel(grayimage, cv2.CV_64F, 1, 0) #grayimage의 경계를 흐리게 하기위해 Sobel X Grediant를 한다.
    sobelY = cv2.Sobel(grayimage, cv2.CV_64F, 0, 1) #grayimage의 경계를 흐리게 하기위해 Sobel Y Grediant를 한다.

    sobelX2 = cv2.Sobel(grayimage2, cv2.CV_64F, 1, 0)   #grayimage2의 경계를 흐리게 하기위해 Sobel X Grediant를 한다.
    sobelY2 = cv2.Sobel(grayimage2, cv2.CV_64F, 0, 1)   #grayimage2의 경계를 흐리게 하기위해 Sobel Y Grediant를 한다.

    sobelX = np.uint8(np.absolute(sobelX))  #Grediant 한 값을 부호없는 정수형으로 바꾼다.
    sobelY = np.uint8(np.absolute(sobelY))  #Grediant 한 값을 부호없는 정수형으로 바꾼다.

    sobelX2 = np.uint8(np.absolute(sobelX2))    #Grediant 한 값을 부호없는 정수형으로 바꾼다.
    sobelY2 = np.uint8(np.absolute(sobelY2))    #Grediant 한 값을 부호없는 정수형으로 바꾼다.

    grayimage = sobelX + sobelY #각각 X,Y로 Grediant 한 값을 더해준다.
    grayimage2 = sobelX2 + sobelY2  #각각 X,Y로 Grediant 한 값을 더해준다.

    # image2에서 0이지만 image1에서 0이 아닌 부분을 0으로 바꾸어 경계를 흐리게해준다.
    row, col = grayimage.shape
    for i in range(0, row):
        for j in range(0,col) :
            if grayimage2[i][j] == 0 and grayimage[i][j] !=0:
                grayimage[i][j] == 0

    # image1에서 image2값을 빼주고 만약 그 값이 음수 라면 부호없는 정수이기 때문에 0으로 바꿔준다.
    for i in range(0, row):
        for j in range(0, col):
            if grayimage[i][j] >= grayimage[i][j] - grayimage2[i][j]:
                grayimage[i][j] = grayimage[i][j] - grayimage2[i][j]

    # 0와 255의 구분을 명확히 하기위해 threshold의 임계값을 계산해준다.
    sum = 0
    for i in range(0, row):
        for j in range(0, col):
            sum = sum + grayimage[i][j]
    sum = (int)(sum / (row * col))
    sum = sum + (int)(sum * 0.1)

    # 알약 중심에 글자가 있기 때문에 GaussianBlur를 통해 중심값을 높여준다.
    grayimage = cv2.GaussianBlur(grayimage, (3, 3), 0)

    # contours를위해 0와 255의 구분을 명확히 하기위해 threshold를 해준다.
    _, th = cv2.threshold(grayimage, sum, 255, 0)

    #findcontours를 이용하여 사진에서 글자별로 박스를 추출한다.
    contours, _ = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #BoundingRect의 가장 큰 값을 구한다.
    max = 0
    for i in range(1, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= max:
            max = w * h
    textarr = []
    origarr = []
    xarr = []

    #contours만큼 반복문을 돌며 글자를 textarr에 input한다.
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

    # x의 순서순으로 배열을 정렬해준다.
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
    #전처리된 알약의 문자를 하나하나 저장해 배열로 return 한다.
    return origarr, textarr
