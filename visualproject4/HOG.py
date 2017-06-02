"""
20143051
kim hyeji

Visual Computing Project 4.
find the HOG feature vector for image
입력 : image
출력 : HOG feature vector
"""
import numpy as np
import cv2

b_size = 16
b_overlap = 1/2
b_shift = int(b_size * b_overlap)
degree = 20

img = cv2.imread('test.png',0)
img = img.astype(np.float32)

#sample size : 134, 70
rows, cols = img.shape
Ix = img.copy()
lx = Ix.astype(np.float32)
Iy = img.copy()
ly = Iy.astype(np.float32)


#gradient 계산
for i in range(1, cols-2):
    Ix[:,i+1] = img[:,i]-img[:,i+2]

for i in range(1, rows-2):
    Iy[i+1,:] = img[i,:]-img[i+2,:]


#orientation
angle = img.copy()
angle = angle.astype(np.float32)

magnitude = img.copy()
magnitude = magnitude.astype(np.float32)

for i in range(rows):
    for j in range(cols):
        if(Ix[i,j] == 0):
            angle[i][j] = 0
        else:
            angle[i][j] = np.degrees(np.arctan(Iy[i,j])/Ix[i,j]) + 90
            magnitude[i][j] = np.sqrt(Ix[i,j]**2 + Iy[i,j]**2)
block = []
#하나의 block의 angle과 magnitude 값을 ang_pat, mag_pat에 저장한다.
for i in range(int(rows/b_shift)):
    for j in range(int(cols/b_shift)):
        #이미지 파일이 b_size의 배수가 아니라서
        #마지막 끝이 맞지 않는 경우
        index_i = b_shift*i
        index_j = b_shift*j
        if(index_i+b_size >= rows and index_j+b_size >= cols):
            mag_pat = magnitude[rows-b_size : rows , cols-b_size:cols]
            ang_pat = angle[rows-b_size : rows , cols-b_size:cols]
        elif(index_i+b_size >= rows):
            mag_pat = magnitude[rows - b_size : rows, index_j : index_j+b_size]
            ang_pat = angle[rows - b_size : rows, index_j : index_j+b_size]
        elif(index_j+b_size >= cols):
            mag_pat = magnitude[index_i : index_i+b_size, cols - b_size:cols]
            ang_pat = angle[index_i : index_i+b_size, cols - b_size:cols]
        else:
            mag_pat = magnitude[index_i : index_i+b_size, index_j: index_j+b_size]
            ang_pat = angle[index_i : index_i+b_size, index_j: index_j+b_size]

        #하나의 block에서 각 셀의 magnitude와 angle 값을 mag, ang 에 저장한다.
        for r in range(2):
            for c in range(2):
                index_r = b_shift*r
                index_c = b_shift*c
                mag = mag_pat[index_r : index_r +b_shift , index_c : index_c +b_shift]
                ang = ang_pat[index_r : index_r +b_shift , index_c : index_c +b_shift]

                #histogram을 제작한다.
                #기여도를 나누어서 배분한다.
                hist = np.zeros(int(180/degree))
                for n_r in range(b_shift):
                    for n_c in range(b_shift):
                        ang_cell = ang[n_r,n_c]
                        ang_num = int(ang[n_r,n_c]/degree)
                        if(ang_num == 0):
                            hist[1] = hist[1] + mag[n_r, n_c] * ((1) * degree - ang_cell) / degree
                            hist[8] = hist[8] + mag[n_r, n_c] * (ang_cell - 8 * degree) / degree
                        elif(ang_num >= 8):
                            hist[8] = hist[8] + mag[n_r, n_c] * ((8) * degree - ang_cell) / degree
                            hist[1] = hist[1] + mag[n_r, n_c] * (ang_cell - 1 * degree) / degree
                        else:
                            hist[ang_num] = hist[ang_num] + mag[n_r,n_c]*((ang_num+1)*degree - ang_cell)/degree
                            hist[ang_num + 1] = hist[ang_num + 1] + mag[n_r,n_c]*(ang_cell - ang_num*degree)/degree

                #normalize using L1-Norm
                hist = hist/np.sqrt(np.linalg.norm(hist)**2+.01)
                block = np.append(block,hist)

#normalize using L2-Norm
block = block/np.sqrt(np.linalg.norm(block)**2+.001)
print(block)