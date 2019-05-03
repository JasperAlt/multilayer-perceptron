import matplotlib.pyplot as plt
from numpy import *
from math import e
import csv

# SETUP

# Parameters
EPOCHS = 50                         # Number of epochs
eta    = 0.1                        # Learning rate
M      = 0.9                        # Momentum
n      = 100                        # Number of hidden neurons

W1 = random.rand(n, 28*28+1) - 0.5  # Weights from input to hidden layer
W2 = random.rand(10, n + 1) - 0.5   # Weights from hidden layer to output

# activation function
def sigmoid(x): return 1 / (1 + e ** (0-x))

# for permuting the training set
def permute(i, l):
  M = array([(i[k],l[k]) for k in range(len(i))])
  P = random.permutation(M)
  return (array([p[0] for p in P]), array([p[1] for p in P]))

test_image = []
test_label = []
train_image = []
train_label = []

# read in data
inp = open("mnist_test.csv")
read = csv.reader(inp, delimiter=",")  # Read in data
for row in read:
  test_label.append(int(row[0]))
  test_image.append([float(x)/255 for x in row[1:]])
inp = open("mnist_train.csv")
read = csv.reader(inp, delimiter=",")
for row in read:
  train_label.append(int(row[0]))
  train_image.append([float(x)/255 for x in row[1:]])
test_image = array(test_image)
test_label = array(test_label)
train_image = array(train_image)
train_label = array(train_label)

test_accuracy = []
train_accuracy = []
tr_cutoff = len(train_image)             # Limit number of examples for testing
te_cutoff = len(test_image)

# MAIN LOOP
for E in range(EPOCHS):
  print(E)

  # SETUP
  P = permute(train_image, train_label)  # permute training data
  train_image = P[0]
  train_label = P[1]
  delta_W2_prev = False
  delta_W1_prev = False
  correct_train = total_train = 0.0
  correct_test = total_test = 0.0

  # TRAINING
  for i in range(tr_cutoff):   # Each example
    #print("Epoch: " + str(E) + " Example: " + str(i))
    X = train_image[i]

    # FORWARD PASS.                   H, O: Hidden and output neuron activations
    H = matmul(W1, append([1],X))     # Append bias and multiply with first set of weights
    H = [sigmoid(h) for h in H]       # activation function
    O = matmul(W2, append([1],H))     # Add another bias, multiply with second weights
    O = [sigmoid(o) for o in O]       # activation function
    g = O.index(max(O))               # pick the winner
    #print("Truth: " + str(train_label[i]) + " Guess: " + str(g) + " " + str(O[g]) + " Right? " + str(train_label[i] == g))
    total_train += 1                  # track for accuracy
    if train_label[i] == g: correct_train += 1

    # BACKWARD PASS
    t = [0.9 if train_label[i] == j else 0.1 for j in range(10)] # set targets
                                      # compute errors
    er_o = [o * (1 - o) * (t[O.index(o)] - o) for o in O]
    er_h = [(H[j] * (1 - H[j]) * sum([W2[k][j] * er_o[k] for k in range(len(O))])) for j in range(len(H))]
                                      # compute deltas
    delta_W2 = eta*outer(er_o, append([1],H)) + M * (delta_W2_prev if delta_W2_prev is not False else array([[0 for k in W2[0]] for j in W2]))
    delta_W1 = eta*outer(er_h,append([1],X)) + M * (delta_W1_prev if delta_W1_prev is not False else array([[0 for k in W1[0]] for j in W1]))

    W2 = W2 + delta_W2                # update weights
    W1 = W1 + delta_W1
    delta_W2_prev = delta_W2          # record deltas
    delta_W1_prev = delta_W1

  #TESTING
  confusion = [[0 for j in range(10)] for i in range(10)]
  for i in range(te_cutoff):   # Each example
    #print("Epoch: " + str(E) + " Example: " + str(i))
    X = test_image[i]
    H = matmul(W1, append([1],X))     # Append bias and multiply with first set of weights
    H = [sigmoid(h) for h in H]       # activation function
    O = matmul(W2, append([1],H))     # Add another bias, multiply with second weights
    O = [sigmoid(o) for o in O]       # activation function
    g = O.index(max(O))

    confusion[g][test_label[i]] += 1
    total_test += 1
    if test_label[i] == g: correct_test += 1
    #print("Truth: " + str(test_label[i]) + " Guess: " + str(g) + " " + str(O[g]) + " Right? " + str(test_label[i] == g))

  # BOOK KEEPING
  train_accuracy += [correct_train/total_train]
  test_accuracy += [correct_test/total_test]

for i in range(10):
  st = ""
  for j in range(10):
    st += "    " + str(confusion[i][j])
  print(st)

# OUTPUT
print(confusion)
plt.plot(test_accuracy, label="Test", color = "#FF5555")
plt.plot(train_accuracy, label = "Training", color = "#770000")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig("mlp_acc_" + str(eta)[2:] + "_" + str(n) + "_" + str(M)[2:]+ "_"+ str(tr_cutoff))
#plt.show()
