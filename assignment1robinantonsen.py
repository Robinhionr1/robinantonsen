import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table = pd.read_csv(r'C:\Users\Robin\repoer\fys2021\SpotifyFeatures.csv')
pop_array  = table[table["genre"] == "Pop"].values #sample size er 9386
classical_array = table[table["genre"] == "Classical"].values #sample size er 9256

#sigmoid funksjonen
def sigmoid(x):
    x = np.array(x)
    return 1/(1+ np.exp(-x))

#deklarerer konstanter
epoker = 1000
learningrate = 0.05
vekter = None
bias = None

#tar x y verdier, altså loudness så liveness
def algoritme(x,y,antall_egenskaper,epoker,vekter,bias,learningrate):
    x = np.array(x)
    antall_epoker, antall_egenskaper = x.shape
    vekter = np.zeros(antall_egenskaper)
    bias = 0
    for _ in range(epoker):
        linear_pred = np.dot(x, vekter) + bias
        predictions = sigmoid(linear_pred)

        dw = (1/antall_epoker) * np.dot(x.T, (predictions-y))
        db = (1/antall_epoker) * np.sum(predictions - y)

        vekter=vekter - learningrate*dw
        bias=bias - learningrate*db
    return vekter, bias

def predict(x,vekter,bias):
    linear_pred = np.dot(x, vekter) + bias
    y_pred = sigmoid(linear_pred)
    class_pred = [0 if y <0.5 else 1 for y in y_pred]
    return class_pred

#LOUDNESS = X
#LIVENESS = Y
#---------------------------------------------------------------------------------------------------------
#lager arrays til alle sangene
pop_songs = pop_array[:,2]
pop_liveness = pop_array[:,11]
pop_loudness = pop_array[:,12]

classical_songs = classical_array[:,2]
classical_liveness = classical_array[:,11]
classical_loudness = classical_array[:,12]

#kombinerer sangene i en felles array
both_songs = np.concatenate((pop_songs, classical_songs))
both_liveness = np.concatenate((pop_liveness, classical_liveness))
both_loudness = np.concatenate((pop_loudness, classical_loudness))

#training sets
train_pop = pop_songs[:7508]
train_classical =  classical_songs[:7404]

train_live_pop = pop_liveness[:7508]
train_live_clas = classical_liveness[:7404]

train_loud_pop = pop_loudness[:7508]
train_loud_clas = classical_loudness[:7404]

#test sets
test_pop = pop_songs[7508:]
test_classical = classical_songs[7404:]

test_live_pop = pop_liveness[7508:]
test_live_clas = classical_liveness[7404:]

test_loud_pop = pop_loudness[7508:]
test_loud_clas = classical_loudness[7404:]

train_both_songs = np.concatenate((train_pop, train_classical))
train_both_liveness = np.concatenate((train_live_pop, train_live_clas))
train_both_loudness = np.concatenate((train_loud_pop, train_loud_clas))
#---------------------------------------------------------------------------------------------------------

#danner training sets og test sets
x_train = train_both_loudness.reshape(-1, 1).astype(float)  
y_train = np.concatenate((np.ones(len(train_pop)), np.zeros(len(train_classical)))) 

x_test = np.concatenate((test_loud_pop, test_loud_clas)).reshape(-1, 1).astype(float) 
y_test = np.concatenate((np.ones(len(test_pop)), np.zeros(len(test_classical)))) 

#caller funksjonen og 
antall_egenskaper = x_train.shape[1] 
vekter, bias = algoritme(x_train, y_train, antall_egenskaper, epoker, vekter, bias, learningrate)

y_pred = predict(x_test, vekter, bias)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__=='__main__':    
    #lager labels for sangene
    y_pop = np.ones(len(pop_songs))
    y_classical = np.zeros(len(classical_songs))  

    #samlede labels
    y = np.concatenate((y_pop, y_classical))

    #plotter
    plt.figure(figsize=(12,6))

    plt.scatter(both_liveness[y == 1], both_loudness[y == 1], label='Pop Songs', color='blue', alpha=0.5)
    plt.scatter(both_liveness[y == 0], both_loudness[y == 0], label='Classical Songs', color='green', alpha=0.5)

    #labels
    plt.xlabel('Loudness')
    plt.ylabel('Liveness')
    plt.show()
