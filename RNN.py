import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

df1 = pd.read_csv("10lacseq.csv")
df2 = pd.read_csv("10lacseqoutput.csv")

print(df1.shape)
print(df2.shape)

#input
x = np.array(df1)

#output
y = np.array(df2)

#print(x)
#print(y)
         
def sigmoid(x):
    return(1/(1+np.exp(-x)))

#Derivative
def der(x):
    return(x*(1-x)) 

# input variables
alpha = 0.1
input_dim = 41
hidden_dim = 29
output_dim = 1
timestep =3
b1=0.63
b2=0.9

#seed
np.random.seed(0)

# initialize and declare neural network weights

syn0 = np.random.uniform(low=-1, high=1,size=(input_dim,hidden_dim)) 
syn1 = np.random.uniform(low=-1, high=1,size=(hidden_dim,output_dim))
synh = np.random.uniform(low=-1, high=1,size=(hidden_dim,hidden_dim)) 

#initializing update wt matrix

syn0_update = np.zeros_like(syn0)
syn1_update = np.zeros_like(syn1)
synh_update = np.zeros_like(synh)

#Training

print('Training')
print('Epoches')
i=0
for i in range(10):
    print(i)
    
    #Resetting the error measure
    overallError = 0
    
    #storing values of synopses
    w0 = syn0
    w1 = syn1
    w2 = synh
    
    #These two lists will keep track of the layer 2 derivatives and layer 1 values at each time step
    l2_deltas = list()
    l1_values = list()
    
    #Time step zero has no previous hidden layer, so we initialize layer_1_values to zero.
    l1_values.append(np.zeros(hidden_dim))
    
    #forwardpass
    for position in range(timestep):
        # hidden layer (input + prev_hidden)
        l1 = sigmoid(np.dot(x,syn0) + np.dot(l1_values[-1],synh)-b1)
        # output layer (new binary representation)
        l2 = sigmoid(np.dot(l1,syn1)-b2)
        
        #calculating gradient values
        l2_error = y-l2
        l2_deltas.append((l2_error)*der(l2))
        overallError += np.abs(l2_error[0])
        
        # store hidden layer so we can use it in the next timestep
        l1_values.append(copy.deepcopy(l1))
    #print(layer_1_values)    
    future_l1_delta = np.zeros(hidden_dim)

    #backwardpass
    for position in reversed(range(timestep)):
        l1 =l1_values[-position-1]
        prev_l1 = l1_values[-position-1]
        
        # error at output layer
        l2_delta = l2_deltas[-position-1]
        #print(prev_layer_1)
        # error at hidden layer
        l1_delta = (future_l1_delta.dot(synh.T) + l2_delta.dot(syn1.T)) * der(l1)
        #print(layer_1_delta.shape)
        
        # let's update all our weights so we can try again
        syn1_update += np.atleast_2d(l1).T.dot(l2_delta)
        synh_update += np.atleast_2d(prev_l1).T.dot(l1_delta)
        syn0_update += x.T.dot(l1_delta)
        future_l1_delta = l1_delta
    
    
    syn0 += syn0_update * alpha
    syn1 += syn1_update * alpha
    synh += synh_update * alpha   
    syn0_update *= 0
    syn1_update *= 0
    synh_update *= 0
#print('error')
#print(overallError)
#print('prediction')    
#print (layer_2)

 #synopses difference
    d0 = abs(syn0-w0)
    d1 = abs(syn1-w1)
    d2 = abs(synh-w2)
    
  #setting condition for stopping the loop
    Flag0=0
    for k in range(len(d0)):
        for l in range(len(d0[k])):
            if not d0[k,l]<15:
                Flag0=1
                break
                
    Flag1=0
    for m in range(len(d1)):
        for n in range(len(d1[m])):
            if not d1[m,n]<15:
                Flag1=1
                break
            
    Flag2=0
    for o in range(len(d2)):
        for p in range(len(d2[o])):
            if not d2[o,p]<15:
                Flag2=1
                break

 #Breaking the loop
    if Flag0==0 and Flag1 and Flag2==0:
        break

#Thresholding
for i in range(len(l2)):
    for j in range(len(l2[i])):
        if l2[i,j]>0.7:
            l2[i,j]=1
        else:
            l2[i,j]=0
            
#print(layer_2) 

#calculating error and accuracy
error=0
accuracy=0
#print('desired    predicted')
for i in range(len(y)):
    for j in range(len(y[i])) :
        if y[i,j] != l2[i,j]:
            error += 1
        else:
            accuracy += 1
            
print('Total number of instances')
print(len(x))
        
print('Error')
print(error)

Error_Percent= (error/len(y))*100
print('Error_Percent')
print(Error_Percent)

print('Accuracy')
print(accuracy)

Accuracy_Percent = (accuracy/len(y))*100
print('Accuracy_Percent')
print(Accuracy_Percent)

#Testing

print('Testing')

df4 = pd.read_csv("2lactesting.csv")
df5 = pd.read_csv("2lactestingoutput1.csv")


#input
x1 = np.array(df4)

#output
y1 = np.array(df5)

#print(yt)
layer_test1_values = list()
layer_test1_values.append(np.zeros(hidden_dim))
for position in range(timestep): 
#hidden layer (input + prev_hidden)
    layer_test1 = sigmoid(np.dot(x1,syn0) + np.dot(layer_test1_values[-1],synh)-b1)
 # output layer (prediction)
    layer_test2 = sigmoid(np.dot(layer_test1,syn1)-b2)
    #save layer 1 
    layer_test1_values.append(copy.deepcopy(layer_test1))
    #print(layer_test2)
    
for i in range(len(y1)):
    for j in range(len(y1[i])):
        if(layer_test2[i][j]<0.59):
            layer_test2[i][j]=0
        if(layer_test2[i][j]>0.59):
            layer_test2[i][j]=1
print('desired   predicted')            

#Calculating error and acuracy
error1=0
accuracy1=0
for i in range(len(y1)):
    for j in range(len(y1[i])) :
        if y1[i,j] != layer_test2[i,j]:
            error1 += 1
        else:
            accuracy1 += 1

print('Total number of instances')
print(len(x1))
        
print('Error')
print(error1)

Error_Percent1= (error1/len(y1))*100
print('Error_Percent')
print(Error_Percent1)

print('Accuracy')
print(accuracy1)

Accuracy_Percent1 = (accuracy1/len(y1))*100
print('Accuracy_Percent')
print(Accuracy_Percent1)

#Calculating True Positives, True Nagative, False Positive and False Negatives
true_positive=0
true_negative=0
false_positive=0
false_negative=0

for i in range(len(y1)):
    for j in range(len(y1[i])) :
        if y1[i,j] == layer_test2[i,j] == 0:
            true_positive += 1
        elif y1[i,j] == layer_test2[i,j] == 1:
            true_negative += 1
        elif y1[i,j]==0 and layer_test2[i,j]==1:
            false_negative += 1
        elif y1[i,j]==1 and layer_test2[i,j]==0:
            false_positive += 1

print('True Positive')             
print(true_positive)  

print('True Negative')
print(true_negative)  

print('False Positive')
print(false_positive)

print('False Negative') 
print(false_negative)  

#Calculating Precision and Recall

Precision=(true_positive/(true_positive+false_positive))*100
print('Precision')
print(Precision)

Recall=(true_positive/(true_positive+false_negative))*100
print('Recall')
print(Recall)

plt.plot(accuracy)
plt.show()