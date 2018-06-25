import numpy as np
from sklearn.svm import LinearSVC
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

X_train = np.array(datasets[0], dtype=np.float32)
X_test = np.array(datasets[1], dtype=np.float32)

y_train = np.array(datasets[2], dtype=np.int32)
y_test = np.array(datasets[3], dtype=np.int32)


print(X_train.shape)
linear_svm = LinearSVC(dual=False).fit(X_train, y_train)
print("Training score: {:.3f}".format(linear_svm.score(X_train, y_train)))
print("Testing score: {:.3f}".format(linear_svm.score(X_test, y_test)))

print(linear_svm.coef_)
print(linear_svm.intercept_)