import h5py
import numpy as np
import cv2
import random

#19600개
with h5py.File('kalph_train.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
h, w = images[0].shape

center =( w/2, h/2 )

#수정한 이미지
image = images[0]
label = labels[0]

for n in range(0,1000):
    image = images[n]
    label = labels[n]

    append_list = np.array([image])
    append_target_list = np.array([label])

    #회전
    for i in [random.randint(-40,40) for i in range(5)]:
        Mat = cv2.getRotationMatrix2D(center, i, 1.0)
        rotated = cv2.warpAffine(image, Mat, (w, h))

        append_list = np.append(append_list,[rotated],axis=0)
        append_target_list = np.append(append_target_list,[label],axis=0)

    #축소/확대
    for i in [random.randint(15, 85) for i in range(5)]:
        default = np.zeros(shape=(h, w))
        # scaling, 축소/확대 별로 보간법이 다르다.
        if i < h:
            resized = cv2.resize(image, (i, i), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (i, i), interpolation=cv2.INTER_LINEAR)
        mid = int(i / 2)
        # 크기를 입력값과 같게 만든다.
        resized = cv2.getRectSubPix(resized, (h, w), (mid, mid))
        append_list = np.append(append_list, [resized], axis=0)
        append_target_list = np.append(append_target_list, [label], axis=0)

    #이동
    for i in [random.randint(0, 15) for i in range(5)]:
        if(i > 5): j = 0
        else:
            j = random.randint(5+i,15+i)

        M= np.float32([[1,0,i],[0,1,j]])
        moved = cv2.warpAffine(image,M, (h,w))

        append_list = np.append(append_list, [moved], axis=0)
        append_target_list = np.append(append_target_list, [label], axis=0)


    #선을 굵게
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(image, kernel, 1)
    images = np.append(images, append_list, axis=0)
    labels = np.append(labels , append_target_list, axis=0)


    a = np.zeros(shape=(h,w))
    #noise
    for _ in range(10):
        s_h = random.randrange(0,w)
        s_w = random.randrange(0,h)
        a[s_h,s_w] = 255
    append_list = np.append(append_list, [image + a], axis=0)
    append_target_list = np.append(append_target_list, [label], axis=0)
    if(n == 1000): break
    if(n%100 == 0):
        print(n)
        print(images.shape)
        print(labels.shape)

    append_list = np.append(append_list, [image + a], axis=0)
    append_target_list = np.append(append_target_list, [label], axis=0)

with h5py.File('kalph_train_add__.hf', 'w') as hf_w:
    hf_w.create_dataset("images",data=images)

    hf_w.create_dataset("labels",data=labels)


