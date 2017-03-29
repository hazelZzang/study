
######################
# linear classified
# visual project
#######################
import numpy as np

#dat[w][x]
def read_data(dat):
    f = open("input.dat", "r")
    line = f.readline()
    dat = [ [[] for j in range(2)] for i in range(3)]
    while line:
        line = int(line) - 1
        for i in range(3):
            dat[i][0].append(float(f.readline().strip()))
            dat[i][1].append(float(f.readline().strip()))
        line = f.readline()
    f.close()
    return np.array(dat)

def weight_vector(w0,w1,w2):
    W = np.array([w0,w1,w2])
    return W

#data_1= [[1,-1],[X1(a), -X1(b)],[X2(a), -X2(b)]]
def data_1(dat, a, b):
    datCol = len(dat[a][0])
    datRow = (len(dat[a]) + 1)
    data_1 = np.ones((datRow, (datCol * 2)))

    data_1[0][datCol:] = -1
    for i in range(1,datRow):
        data_1[i] = np.append(dat[a][i-1],dat[b][i-1] * -1.0)
    return data_1

def forward(W, D, learningLate):
    while(1):
        a = np.dot(W, D) < 0
        print(np.dot(W,D))
        if a.sum() == 0: break;
        J = np.sum(D[:,a], axis = 1) * learningLate
        J[0] = 0
        W = W + J
        print("가중치")
        print(W)

    return W


def relaxation(W, D, learningLate, b):
    while(1):
        dot_WD = np.dot(W, D)
        print(b - dot_WD)
        a = b - dot_WD > 0
        if a.sum() == 0: break;

        S = ((b - dot_WD) / ( np.sum(D * D, axis=0) - 1) ) #열끼리 합
        J = (np.sum(S[a] * D[:,a], axis = 1) * learningLate)
        W = W + J
    return W


def LMS(W, D, learningLate, b, theta):
    while(1):
        dot_WD = np.dot(W, D)
        J = (b - dot_WD) * D * learningLate
        a = np.sum(J,axis = 0) < theta
        if a.sum() == 0: break;
        print(a)
        print(a.sum())
        W = W + np.sum(J, axis = 1)
        D = D[:,a]
        print(W)
    return W


dat = []
dat = read_data(dat)

"""
#####1번
W1 = weight_vector(0.5,-0.5,0.5)
D1 = data_1(dat,0,1)
L1 = 0.01
print(forward(W1,D1,L1))
"""
"""
#####2번
W2 = weight_vector(0,0,0)
D2 = data_1(dat,0,2)
L2 = 0.01
relaxation(W2,D2,L2,0.5)
"""
######3번
W3 = weight_vector(0,0,0)
D3 = data_1(dat,0,2)
L3 = 0.01
margin = 0.1
threshold = 1
LMS(W3,D3,L3,margin,threshold)


