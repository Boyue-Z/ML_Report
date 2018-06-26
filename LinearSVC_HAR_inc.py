import numpy as np
from sklearn.svm import LinearSVC
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

count = 0


need_idx = []
sizeof_feature = 561

training_scores = []
testing_scores = []


# for idx in need_idx:
#     print(idx)
for sizeof in range(1, sizeof_feature + 1):
    print("Level: ", sizeof)
    # init
    datasets = [[],[],[],[]]
    count = 0

    # inc sizeof range
    need_idx.append(int(sizeof))        
    
    # rotiune for reading dataset
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

    # print("Size of :", len(need_idx))
    # print("X_Train shape")
    # print(X_train.shape)
    # print("X_Test shape")
    # print(X_test.shape)

    # print("Y_Train shape")
    # print(y_train.shape)
    # print("Y_Test shape")
    # print(y_test.shape)

    linear_svm = LinearSVC().fit(X_train, y_train)
    training_score = linear_svm.score(X_train, y_train)
    testing_score = linear_svm.score(X_test, y_test)
    # print("Training score: {:.3f}".format(training_score))
    # print("Testing score: {:.3f}".format(testing_score))
    
    training_scores.append(training_score)
    testing_scores.append(testing_score)

    # clear datatsets
    datasets.clear()

    # print()

with open("result.txt", "w") as f:
    json.dump([need_idx, training_scores, testing_scores], f)

plt.plot(need_idx, training_scores, color='blue')
plt.plot(need_idx, testing_scores, color='red')

plt.show()