import numpy as np
from sklearn.svm import SVC

# features
F_FILES = [
    'X_train.txt', 'X_test.txt'
]

# outcome 
O_FILES = [
    'y_train.txt', 'y_test.txt'
]

datasets = [[],[],[],[]]
count = 0

for filename in F_FILES:
    f = open(filename, "r")
    for line in f:
        datasets[count].append([float(number) for number in line.split()])
    count += 1
    f.close()

for filename in O_FILES:
    f = open(filename, "r")
    for line in f:
        datasets[count].extend([int(number) for number in line.split()])
    count += 1
    f.close()

X_train = datasets[0]
X_test = datasets[1]

y_train = datasets[2]
y_test = datasets[3]

svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X_train, y_train)
print("Training score: {:.3f}".format(svm.score(X_train, y_train)))
print("Testing score: {:.3f}".format(svm.score(X_test, y_test)))