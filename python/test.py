from char_test import char_test
from shape_test import shape_test
import cv2
import json
import sys

from image_processing import image_processing
from make_square import make_square
from findtext import findtext

def char_decoding(c):
    if c < 10: # Digit
        return str(chr(c+48))
    elif c == 36: # default
        return ""
    else: # English
        return str(chr(c - 10 + 65))

def shape_decoding(s):
    if s == 0:
        return "circle"
    elif s == 1:
        return "ellipse"
    elif s == 2:
        return "etc"
    else:
        return "Wrong!"

image = cv2.imread(sys.argv[1])


image1 = cv2.resize(image, (420, 560), interpolation=cv2.INTER_LINEAR_EXACT)
imgcolorc,imgcolorf,imgshape,colorname = image_processing(image1)
_,thres = findtext(imgcolorc,imgcolorf)

c_arr = ''
for th in thres:
    th_sq = make_square(th)

    c = char_test(th_sq)[0]
    
    c_arr = c_arr + char_decoding(c)

s = shape_test(imgshape)[0]

s_arr = shape_decoding(s)


result = {'character' : c_arr, 'shape' : s_arr, 'color' : colorname}

data = json.dumps(result)
print(data)


