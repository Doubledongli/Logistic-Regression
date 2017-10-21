from numpy import *
import logRegres
import logRegresGo

dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr,labelMat)
logRegres.plotBestFit(weights)
