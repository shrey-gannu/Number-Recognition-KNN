import cv2
import numpy as np
import scipy
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

############## reading images ###########################
train = cv2.imread('digits.png',cv2.IMREAD_GRAYSCALE)
testdigits = cv2.imread('test_digits.png',cv2.IMREAD_GRAYSCALE)

############## converting 3-d numpy.ndarray into 2d lists ###########################

def three_D_converter(digits):
    rows = np.vsplit(digits,50)
    cells =[]
    actualcell = []
    for row in rows:
        cols = np.hsplit(row,50)
        for lists in cols:
            for cell in lists:
                for i in cell:
                    cells.append(i)
            actualcell.append(cells)
            cells = []
    return actualcell



def two_D_converter(digits):
    cols = np.vsplit(testdigits,50)
    cells =[]
    actualcell = []
    for lists in cols:
        for cell in lists:
            for i in cell:
                cells.append(i)
        actualcell.append(cells)
        cells = []
    return actualcell

X_train = three_D_converter(train)
X_test = two_D_converter(testdigits)


##### training ##############


Y = np.arange(10)
y = np.repeat(Y,250)
y_expect = np.repeat(Y,5)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y)

##### testing ##################


y_pred = clf.predict(X_test)
print(y_pred)

########## calculating accuracy ################3


print(metrics.classification_report(y_expect,y_pred))
cv2.imshow(' 0',testdigits)
cv2.waitKey(0)
cv2.destroyAllWindows
