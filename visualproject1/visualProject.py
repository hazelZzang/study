###############################################
#Kookmin University
#Latest technology in visual computing Class
#20143051 kim hye ji
#Lab1.
###############################################

from __future__ import division
from matplotlib import pyplot as plt
import numpy
import numpy.linalg as lin
from numpy import meshgrid
from numpy import arange

"""
DATA[CLASS][FEATURE] = [f1_1,f1_2 ...]
TEST[CLASS] = [f1,f2,f3,f4]
"""
CLASSIFY = 3
FEATURE = 4

data = [[0] * 4 for x in range(CLASSIFY)]
test = [[] for x in range(CLASSIFY)]

###################
# File Open
##################
def fileOpen():
    f = open("Iris_train.dat", 'r')
    line = f.readlines()
    for l in line:
        word = l.split()
        CLASS = int(word[-1]) - 1
        for feature in range(FEATURE):
            if data[CLASS][feature] == 0:
                data[CLASS][feature] = []
            data[CLASS][feature].append(float(word[feature]))
    f.close()

def testfileOpen():
    f = open("Iris_test.dat", 'r')
    line = f.readlines()
    for l in line:
        word = l.split()
        CLASS = int(word[-1]) - 1
        test[CLASS].append([float(i) for i in word[0:FEATURE]])
    f.close()

###################
# calculate
##################
def Dot(v, w):
    return sum(v_i * w_i
               for v_i,w_i in zip(v,w))

def Mean(list):
    return sum(list)/len(list)

def DeMean(c, f):
    mean = meanData[c][f]
    return [i - mean
            for i in data[c][f]]

def Covariance(c, f1, f2):
    return Dot(DeMean(c,f1),DeMean(c,f2)) / (len(data[c][f1]) - 1)

def CovalMatrix(i,rowcol):    #covariance about class
    return [[round(Covariance(i, f1, f2), 2)
             for f2 in range(rowcol)] for f1 in range(rowcol)]

def MeanMatrix():
	return [[round(Mean(data[x][y]),2) 
			for y in range(FEATURE)] for x in range(CLASSIFY)]

def ReverseMatrix(C):
    return lin.inv(C)

def minusMatrix(a,b):
    return [a_i - b_i
            for a_i in a for b_i in b]



###################
# Discriminant function
##################
def DiscriFunc(testData, covar_, mean_):      # g(x) value (normal distribution) between testData and class
    data = numpy.array(testData) #1XM
    func = DiscriFuncClass(covar_,mean_)
    return Dot(Dot(data,func[2]),data.T) + Dot(func[1], data.T) + func[0]

def DiscriFuncClass(covar_, mean_):         # g(x) value with Class
    CMat = numpy.array(covar_) #MXM
    RCMat = numpy.array(ReverseMatrix(CMat))
    mean = numpy.array(mean_) # 1XM

    func = []
    #constant
    func.append(Dot(Dot((-1/2) * mean,RCMat) ,mean.T) + (-1/2)*lin.det(CMat))
    #simple
    func.append(Dot(RCMat,mean))
    #quadratic
    func.append((-1/2) * RCMat)
    return func

def DiscriFuncCompare(covar1, covar2, mean1, mean2):    #Discriminant function between two Classes
    func = []
    func1 = DiscriFuncClass(covar1,mean1)
    func2 = DiscriFuncClass(covar2,mean2)

    for i in range(CLASSIFY):
        func.append(func1[i] - func2[i])
    return func

def DiscriFuncTest(testData, covar1, covar2,mean1,mean2):    #Discriminant function Test
    data = numpy.array(testData) #1X4
    func = DiscriFuncCompare(covar1,covar2, mean1,mean2)
    return Dot(Dot(data,func[2]),data.T) + Dot(func[1], data.T) + func[0]

###########################
# Confusion Matrix
###########################
def Classify(rowcol):
    cMat = [[0] * CLASSIFY for x in range(CLASSIFY)]
    for i in range(CLASSIFY):
        for j in range(len(test[i])):
            if(rowcol == 4):
                clList = [DiscriFunc(test[i][j], CovalMatrix(k,4), meanData[k]) for k in range(CLASSIFY)]
            elif(rowcol == 2):
                clList = [DiscriFunc(test[i][j][:2], CovalMatrix(k, 2), meanData[k][:2]) for k in range(CLASSIFY)]
            result = (clList.index(max(clList)))
            cMat[i][result] += 1

    return cMat


fileOpen()
testfileOpen()

meanData = MeanMatrix()

#classified discriminant function
print ('Problem1 confusion matrix')
print(Classify(4))

########## Problem.2

def mahalDist(testData, c):
    CoMat = numpy.array(CovalMatrix(c,2))
    X = numpy.array(minusMatrix(testData,meanData[c][:2]))
    return numpy.sqrt(Dot(Dot(X,ReverseMatrix(CoMat)),X.T))

##############################
# Ploting
##############################
def printDataPlot2X2():         #training data plot
    colorMat = ['red', 'black', 'green']

    for i in range(CLASSIFY):
        plt.scatter(data[i][0], data[i][1], color = colorMat[i])
        plt.scatter(meanData[i][0], meanData[i][1], marker='+', color = colorMat[i])


def printTestPlot2X2():
    colorMat = ['red', 'black', 'green']
    for i in range(len(test[0])):
        for j in range(CLASSIFY):
            plt.scatter(test[j][i][0], test[j][i][1], edgecolors = colorMat[j], facecolors='none' )

def mahalDistPlot(dist):
    colorArr = ['red', 'black', 'green']
    for cNum in range(CLASSIFY):
        rangeX = arange(4.0, 8.0, 0.025)
        rangeY = arange(1.0, 5.0, 0.025)

        X, Y = meshgrid(rangeX, rangeY)

        RCoMat = ReverseMatrix(CovalMatrix(cNum,2))
        a = RCoMat[0][0]
        b = RCoMat[0][1]
        c = RCoMat[1][0]
        d = RCoMat[1][1]
        m1 = meanData[cNum][0]
        m2 = meanData[cNum][1]
        eq = (a * (X-m1) + c * (Y-m2)) * (X-m1) + (b * (X - m1) + d * (Y - m2)) * (Y-m2) - dist
        plt.contour(X, Y, eq, [0], colors=colorArr[cNum])

def DeciBoundaryPlot(c1, c2, m1, m2, color = 'black', rangeChange = 0):
    if(rangeChange == 1):
        rangeX = arange(6.0, 8.0, 0.025)
    else:
        rangeX = arange(4.0, 8.0, 0.025)
    rangeY = arange(1.0, 5.0, 0.025)

    X, Y = meshgrid(rangeX, rangeY)
    coefMat = (DiscriFuncCompare(c1,c2,m1,m2))

    #quadratic coefficient(2X2 [a b c d])
    a = coefMat[2][0][0]
    b = coefMat[2][0][1]
    c = coefMat[2][1][0]
    d = coefMat[2][1][1]
    #simple coefficient(1X2 [e f])
    e = coefMat[1][0]
    f = coefMat[1][1]
    #constant
    g = coefMat[0]
    testeq = X * (a*X+c*Y+e) + Y * (b*X+d*Y+f)+g
    plt.contour(X, Y, testeq,[0], colors=color)
    return lambda X,Y: X * (a*X+c*Y+e) + Y * (b*X+d*Y+f)+g

def testDataPlot(func01, func12, func20):
    colorMat = ['red', 'black', 'green']

    for c in range(3):
        for f in range(len(test[c])):
            x = test[c][f][0]
            y = test[c][f][1]
            if(func01(x,y) > 0):
                if(func20(x,y) > 0):
                    result = 2
                else:
                    result = 0
            elif(func12(x,y) > 0):
                result = 1
            else:
                result = 2

            if(result != c):
                plt.scatter(test[c][f][0], test[c][f][1], edgecolors=colorMat[c], facecolors='yellow')
            else:
                plt.scatter(test[c][f][0], test[c][f][1], edgecolors=colorMat[c], facecolors='none')


CoMat2X2 = [CovalMatrix(i, 2) for i in range(CLASSIFY)]

#data print
printDataPlot2X2()

#Mahalanobis distance print
mahalDistPlot(2)

#decision boundary print
DBfunc01 = DeciBoundaryPlot(CoMat2X2[0],CoMat2X2[1], meanData[0][:2], meanData[1][:2],'green')
DBfunc12 = DeciBoundaryPlot(CoMat2X2[1],CoMat2X2[2], meanData[1][:2], meanData[2][:2],'blue',1)
DBfunc20 = DeciBoundaryPlot(CoMat2X2[2],CoMat2X2[0], meanData[2][:2], meanData[0][:2],'black')

#classified using decision boundary, plot
testDataPlot(DBfunc01,DBfunc12,DBfunc20)

#classified discriminant function
print ('Problem2 confusion matrix')
print(Classify(2))

plt.show()




"""
for i in range(10):
    print(mahalDist(test[0][i][0:2], 0))
"""