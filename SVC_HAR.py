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


need_idx = list(range(1, 562))
# for idx in need_idx:
#     print(idx)

for filename in F_FILES:
    f = open(filename, "r")
    for line in f:
        numbers = []
        number_idx = 1
        for number in line.split():
            if number_idx in need_idx:
                numbers.append(float(number))
            number_idx += 1            
        datasets[count].append(numbers)
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
print(X_test.shape)

svm = SVC(kernel='rbf', C=1000, gamma=10).fit(X_train, y_train)
print("Training score: {:.3f}".format(svm.score(X_train, y_train)))
print("Testing score: {:.3f}".format(svm.score(X_test, y_test)))