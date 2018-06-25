import numpy as np
from sklearn.neural_network import MLPClassifier

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


# Test 1 ReLU + adam
# mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
#                     solver='adam', verbose=10, tol=1e-4, random_state=1)

# Test 2 logistic + adam
# mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
#                     activation = 'logistic', solver='adam', verbose=10, tol=1e-4, random_state=1)

# Test 3 ReLU + SGD
mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=400, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))