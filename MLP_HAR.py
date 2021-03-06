import numpy as np
from sklearn.neural_network import MLPClassifier
import numpy as np

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

# Time Domain
need_idx = list(range(1, 10))
need_idx.append(16)
need_idx.extend(range(20, 26))
need_idx.extend(range(38, 41))

need_idx.extend(range(41, 50))
need_idx.append(56)
need_idx.extend(range(60, 66))
need_idx.extend(range(78, 81))


# Freq Domain
need_idx.extend(range(282, 285))
need_idx.extend(range(288, 291))

need_idx.extend(range(440, 443))
need_idx.extend(range(446, 449))

print(len(need_idx))

for idx in need_idx:
    print(idx)



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


# Test 1 ReLU + adam
# mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
#                     solver='adam', verbose=10, tol=1e-4, random_state=1)

# Test 2 logistic + adam
# mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=1e-4,
#                     activation = 'logistic', solver='adam', verbose=10, tol=1e-5, random_state=1)

# Test 3 ReLU + SGD
mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-6, random_state=1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))