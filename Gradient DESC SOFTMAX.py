'''

This Example shows the Mnist Data Set classification using Gradient Descent Algorithm which was
coded in Python

If You like my work please consider supoorting me by following me on github and Youtube

Thanking you,
Soumil Nitin Shah

Bachelor in Electronic Engineering
Master in Electrical Engineering
Master in Computer Engineering

Graduate Teaching/Research Assistant

Python Developer

soushah@my.bridgeport.edu
——————————————————----------------------------------
Linkedin:	https://www.linkedin.com/in/shah-soumil

Github
https://github.com/soumilshah1995

Youtube channel
https://www.youtube.com/channel/UC_eOodxvwS_H7x2uLQa-svw

------------------------------------------------------
'''

# Import library
try:
    import os
    import sys
    import cv2
    import numpy as np
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    import time
    import datetime

except:
    print("Library not found ")

now_time= datetime.datetime.now()

counter = 0
x_data_epoch = []
y_data_error = []

train = np.empty((1000,28,28), dtype='float64')
trainY = np.zeros((1000,10,1))

test = np.empty((10000, 28, 28), dtype='float64')
testY = np.zeros((10000, 10, 1))

# --------------------------------------------------------Load in Image--------------------------------------
i = 0
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000'):
    y = int(filename[0])

    trainY[i, y] = 1.0

    train[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000/{0}'.format(filename), 0)/255.0
    i = i+1

# -------------------------------------------------LOAD TEST IMAGE ------------------------------------------------

i = 0
# read test data
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000'):
    y = int(filename[0])

    testY[i, y] = 1.0

    test[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000/{0}'.format(filename), 0)/255.0
    i = i+1
# ---------------------------------------------------------------------------------------------------------------------

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2], 1)
testX = test.reshape(test.shape[0], test.shape[1] * test.shape[2], 1)

# ---------------------------------  Neural Network ---------------------------------------
numNeuronsLayer1 = 100       # Number of Neuron in Layer 1
numNeuronsLayer2 = 10        # Expected output
numEpochs = 50              # Epoch
learningRate = 0.2          # Learning Rate

w1 = np.random.uniform(low=-0.1,high=0.1,size=(numNeuronsLayer1,784))
b1 = np.random.uniform(low=-1,high=1,size=(numNeuronsLayer1, 1))
w2 = np.random.uniform(low=- 0.1,high=0.1,size=(numNeuronsLayer2, numNeuronsLayer1))
b2 = np.random.uniform(low=-0.1,high=0.1,size=(numNeuronsLayer2,1))

for n in range(0, numEpochs):

    loss = 0

    trainX, trainY = shuffle(trainX, trainY)

    gradw2 = np.zeros((numNeuronsLayer2, numNeuronsLayer1))
    gradw1 = np.zeros((numNeuronsLayer1, 784))
    gradb1 = np.zeros((numNeuronsLayer1, 1))
    gradb2 = np.zeros((numNeuronsLayer2, 1))

    for i in range(trainX.shape[0]):

        s1 = np.dot(w1,trainX[i]) + b1          # S1 = W1.X + B1
        a1 = 1 / (1 + np.exp(-s1))              # A1 = 1/ 1 + exp(-s1)

        s2 = np.dot(w2, a1) + b2                # S2 = W2.A1+ B1
        a2 = np.exp(s2) / np.exp(s2).sum()

        #loss += (0.5 * ((a2-trainY[i])*(a2-trainY[i]))).sum()
        loss = - np.sum(trainY[i] * np.log(a2))

        # -------------------------------------- BACK PROPOGATE --------------------------------------

        delta2 = a2 - trainY[i]

        derv_act_1 =np.multiply(a1 , (1 - a1))                  # derivative of act = A1.(1-A1)
        error_2 = np.dot(w2.T, delta2)                          # e_2 = Delta2 . W2
        delta1 = np.multiply( error_2, derv_act_1)              # Delta1 = Delta2 . A1(1-A1).W2

        gradw2 += np.dot(delta2,a1.T)                           # Grad = Delta.Output of previous Layer
        gradw1 += np.dot(delta1, trainX[i].T)                   # Grad = Delta.Output of previous Layer
        gradb2 += delta2                                        # GradB2 = Delta2
        gradb1 += delta1                                        # GradB1 = Delta1

    w2 = w2 - learningRate * (gradw2/1000)                      # Accumulate all Gradient and Divide by 1000
    b2 = b2 - learningRate * gradb2/1000                        # Accumulate all Gradient and Divide by 1000

    w1 = w1 - learningRate * (gradw1/1000)                       # Accumulate all Gradient and Divide by 1000
    b1 = b1 - learningRate * (gradb1/1000)                       # Accumulate all Gradient and Divide by 1000

    x_data_epoch.append(n)                                      # Append error and loss
    y_data_error.append(loss)

    print("epoch = " + str(n) + " loss = " + (str(loss)))

print("done training , starting testing..")
accuracyCount = 0

# --------------------- Testing Data ------------------------------------------------

for i in range(testY.shape[0]):

    s1 = np.dot(w1,testX[i]) + b1           # S1 = W.X + B1
    a1 = 1/(1+np.exp(-1*s1))                # A1 = 1/ 1 + exp(-S1)

    s2 = np.dot(w2,a1) + b2                 # S2 = W2.A1 + B2
    a2 = 1/(1+np.exp(-1*s2))                # A2 = 1/ 1 + exp(-S2)

    a2index = a2.argmax(axis=0)             # Select Max from 10 Neuron

    if testY[i, a2index] == 1:              # calculate The Accuracy
        accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/10000.0))


end = datetime.datetime.now()
t = end-now_time

# Plot The Graph

print('time{}'.format(t))
plt.plot(x_data_epoch,y_data_error)
plt.xlabel('X axis Neuron {}'.format(numNeuronsLayer1))
plt.ylabel('Loss')

plt.title('\n Gradient Descent \n Execution Time:{} \n Accuracy Count {}\n Loss :{}'.format(t,accuracyCount,loss))
plt.show()