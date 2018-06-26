import numpy as np
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import matplotlib.pyplot as plt

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
sizeof_feature = 561

training_scores = []
testing_scores = []

need_idx = list(range(1, sizeof_feature + 1))
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


for idx in range(1, sizeof_feature + 1):
    X_train_sub = np.delete(X_train, range(idx,sizeof_feature + 1), axis=1)
    X_test_sub = np.delete(X_test, range(idx,sizeof_feature + 1), axis=1)
    print("Level:", idx)
    print(X_train_sub.shape)
    print(X_test_sub.shape)

    # Test 3 ReLU + SGD
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1)

    mlp.fit(X_train_sub, y_train)
    training_score = mlp.score(X_train_sub, y_train)
    testing_score = mlp.score(X_test_sub, y_test)

    print("Training score: {:.3f}".format(training_score))
    print("Testing score: {:.3f}".format(testing_score))
    
    training_scores.append(training_score)
    testing_scores.append(testing_score)

    print()

with open("result_MLP.txt", "w") as f:
    json.dump([need_idx, training_scores, testing_scores], f)

plt.plot(need_idx, training_scores, color='blue')
plt.plot(need_idx, testing_scores, color='red')

plt.show()