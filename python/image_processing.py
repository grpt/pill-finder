# -*- coding: utf-8 -*-

import cv2
import numpy as np

 #이미지에서 배경부분을 지우기 위한 함수
def backproject(source, target, levels=2, scale=1):
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv2.calcHist([hsv], [0, 1], None, [levels, levels], [0, 180, 0, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], scale)
    return dst

#알약의 글씨를 지우는 함수
def text(img):              
    imgx = img

    imglast = imgx.copy()

    row = 0
    col = 0
    row, col, _ = imgx.shape

    cntb = 0
    cntf = 0

    #kernel size 3으로 설정

    kernel = np.ones((3, 3), np.uint8)
    imgx1 = cv2.morphologyEx(imgx, cv2.MORPH_CLOSE, kernel)

    #kernel size 증가하면서 전체 크기 변할시 그 전 값 리턴

    for ksize in range(3, row, 2):
        kernel = np.ones((ksize, ksize), np.uint8)


        #kernel을 이동해가며 kernel size 안의 픽셀을 검사해 더 많은 픽셀의 bgr값으로 나머지 픽셀을 수정
        #약의 글씨부분을 약의 색깔과 비슷하게 만듬

        imgx2 = cv2.morphologyEx(imgx, cv2.MORPH_CLOSE, kernel)     
        b, g, r = cv2.split(imgx2)

        #morphologyEx함수를 적용한 이미지의 알약부분 픽셀수를 셈
        for i in range(int(row / 2)):
            for j in range(int(col / 2)):
                if b[i][j] + g[i][j] + r[i][j] > 30:    
                    cntf += 1

        #이전의 이미지의 픽셀수와 비교하여 알약의 이미지가 원본보다 커졌을때
        #글씨가 충분히 지워졌을거라 판단하고 리턴

        if cntb is 0:
            cntb = cntf                                 
        elif cntf - cntb > int(cntb * 0.02):            
            break
        imgx1 = imgx2
        cntf = 0

    return imgx1


def fill(imgx, imglast, threshMap, img_size): #이미지에서 알약부분의 빈곳을 채우는 함수
    row, col = img_size
    kernel = np.ones((3, 3), np.uint8)


    #kernel을 이동해가며 kernel size 안의 픽셀을 검사해 더 많은 픽셀의 bgr값으로 나머지 픽셀을 수정
    #알약부분에 빈공간이 일부분 채워지게 된다
    #이미지에서 알약부분의 중앙 최상단, 중앙 최좌단의 좌표를 i1과 j1에 저장한다.
    threshMap1 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)   
                                                                        
    for i1 in range(row):                                               
        if threshMap1[i1][(int)(col / 2)] > 20:
            break
    for j1 in range(col):
        if threshMap1[(int)(row / 2)][j1] > 20:
            break
    size = 5
    while (size < 20):       

        #kenel size를 증가시키면서 morphologyEx함수를 적용시키고
        kernel = np.ones((size, size), np.uint8)
        threshMap2 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)

        size += 2
        for i2 in range(row):
            if threshMap2[i2][(int)(col / 2)] > 20:
                break
        for j2 in range(col):
            if threshMap2[(int)(row / 2)][j2] > 20:
                break
        #이미지의 알약 크기가 변동하면 채우지 않아야 하는 부분을 채웠으므로 그전의 백업해두었던 이미지를 밑의 과정에서 사용한다.
        if i1 != i2 or j1 != j2:                                    
            break
        else:
            threshMap1 = threshMap2
            i1 = i2
            j1 = j2

    threshMap = threshMap1
    #GaussianBlur를 적용
    threshMapfilter = cv2.GaussianBlur(threshMap, (7, 7), 0)    
    imglast2 = imglast.copy()
    #threshMap의 이미지에서 알약부분을 검출하였으므로 원본 이미지에 동일좌표를 적용하여 원본 이미지에서 알약부분을 검출한다.
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

#bgr채널의 이미지를 hsv채널로 변경하는 함수
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

#알약의 색을 검출하는 함수
def color(bbb):                     
    b, g, r = cv2.split(bbb)
    cb = 0
    cg = 0
    cr = 0
    cnt = 0
    h, w, _ = bbb.shape
    for i in range(h):
        for j in range(w):
            #알약부분의 픽셀들의 평균 bgr을 구한다.
            if (b[i][j] != 0) and (g[i][j] != 0) and (r[i][j] != 0):            
                cb += b[i][j]
                cg += g[i][j]
                cr += r[i][j]
                cnt += 1
    cb = (int)(cb / cnt)
    cg = (int)(cg / cnt)
    cr = (int)(cr / cnt)
  
    #bgr 평균값을 hsv로 변환한다
    #hsv 값을 이용하여 색을 판단
    h, _, v = rgb2hsv(cr, cg, cb)                   
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
    imgx = img
    imglast = imgx.copy()
    #이미지의 노이즈를 pyrMeanShiftFiltering적용하여 제거한다
    cv2.pyrMeanShiftFiltering(imgx, 2, 10, imgx, 4)         

    #이미지에 배경을 제거하기 위한 객체를 생성.
    backproj = np.uint8(backproject(imgx, imgx, levels=2))  
    backproj = 255 - backproj


    #히스토그램 균일화를 통해 배경을 제거
    cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX) 

    #saliencyMap을 이용하여 특이점을 검출한다.
    saliencies = [backproj, backproj, backproj]             
    saliency = cv2.merge(saliencies)

    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)

    saliencyMap = saliency

    #threshold를 적용하여 특이점 이외의 부분을 제거
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


    #알약의 경계를 findContours통해 찾음
    contours, hierarchy = cv2.findContours(threshMapfilter * 1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     

    #이미지의 경계들의 좌표를 크기순으로 정렬
    contours = sorted(contours, key=cv2.contourArea)                        

    # ------------------------------크기 정사각형 만드는 부분------------------------------------------
    flag = 0
    size = len(contours) -1
    for j in range(0,1000):
        for i in range(-1, size): 
            #이미지에서 알약부분은 화면 중앙에 가깝게 위치해있으므로 위에서 찾은 경계들중 중앙에 가까운 경계를 찾음                                                      
            x, y, w, h = cv2.boundingRect(contours[i])
            if (x - j*10< 210 and x + w  + j*10> 210 and y - j*10< 280 and y + h + j*10> 280):
                flag =1
                break
        if flag == 1:
            break
    #찾은 경계부분을 이용 원본 이미지로부터 최종 알약부분을 검출
    #찾은 알약의 이미지는 추후에 정사각형의 사이즈로 사용되기때문에 이미지를 정사각형으로 만들어줌
    imgcolorc = imglast[y:y + h, x:x + w]               
    if (h > w):                                         
        x = x - (int)((h - w) / 2)
        w = h
    else:
        y = y - (int)((w - h) / 2)
        h = w
    # ------------------------------------------------------------------------------

    imgshape = threshMapfilter[y:y + h, x:x + w]

    #알약의 색을 검출
    imgcolorc, colorname = color(imgcolorc)

    #글자검출에 필요한 이미지(글자부분을 지운 알약이미지)를 생성         
    imgcolorf = text(imgcolorc)                     

    #알약 이미지, 글자검출을 위한 이미지, 알약의 모양 이미지, 알약의 색 리턴
    return imgcolorc, imgcolorf, imgshape, colorname    