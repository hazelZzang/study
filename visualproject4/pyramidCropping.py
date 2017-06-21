import cv2
import numpy as np

test_file = './test3.png'
row = 134
col = 70
shift = 0.5

shift_row = int(shift * row)
shift_col = int(shift * col)

def crop_image(file):
    img_crop = []
    img_locate = []
    img = cv2.imread(file, 0)

    #다양한 크기의 피라미드 이미지를 만든다.
    for i in (0.4, 0.6, 0.8 ,1):
        re_img = cv2.resize(img, None, fx=i,fy=i, interpolation=cv2.INTER_AREA)

    #shift만큼 이동하며 이미지를 잘라낸다.
        r, c = re_img.shape
        r -= row
        c -= col
        if(r < 0 or c < 0): continue
        #이미지 이동 시작 점들
        r_range = [x*shift_row for x in range(int(r/ shift_row + 1))]
        c_range = [y*shift_col for y in range(int(c/ shift_col + 1))]

        for x in r_range:
            for y in c_range:
                crop = re_img[x:x+row, y:y+col]
                img_crop.append(crop/255.0)
                img_locate.append([x,y,i])
    return np.array(img_crop), np.array(img_locate)

