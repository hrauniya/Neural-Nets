"""
@author:Harsha Rauniyar
implementing the neural network algorithm
"""

from enum import unique
import sys
import random
import math
import numpy as np
import pandas as pd
import csv

pd.options.mode.chained_assignment = None 

#handling command line arguments to be entered in the terminal
number_neurons=int(sys.argv[2])
learning_rate=float(sys.argv[3])
training_percentage=float(sys.argv[4])
random_seed= int(sys.argv[5])
threshold=float(sys.argv[6])
dataset=sys.argv[1]

random.seed(random_seed)

#initialize dataframe
dataframe=pd.read_csv(sys.argv[1])

dataframe=dataframe.sample(random_state=random_seed, frac=1)


#spliting the dataset to training, and test depending on the percentage given by the user
dataframe_length = len(dataframe.index)
training_set_length=round(training_percentage*dataframe_length)
validation_set_length=round(((1-training_percentage)/2)*dataframe_length)

training_df = dataframe.iloc[0:training_set_length]
validation_df = dataframe.iloc[training_set_length:training_set_length+validation_set_length]
test_df = dataframe.iloc[training_set_length+validation_set_length:]


#scaling functions
def scaling(col):
    scaled_value = (col - col.min()) / (col.max() - col.min())
    return scaled_value

def scaling2(col, name):
    scaled_value = (col - max_min[name][0]) / (max_min[name][1] - max_min[name][0])
    return scaled_value

max_min = {}

#normalizing data
axis=1
for column in training_df.columns[1:]:
    
    unique_list = training_df[column].unique()    
    
    if isinstance(training_df[column].tolist()[0], (float, int)) and len(unique_list)>1:
        
        max_min[column] = (training_df[column].min(), training_df[column].max())
        training_df[column] = scaling(training_df[column])
        
    
    if isinstance(validation_df[column].tolist()[0], (float, int)) and len(unique_list)>1:
        validation_df[column] = scaling2(validation_df[column], column)
    
    if isinstance(test_df[column].tolist()[0], (float, int)) and len(unique_list)>1:
        test_df[column] = scaling2(test_df[column], column)

    
    else:
        
        unique_list = training_df[column].unique()
        
        dummies = pd.get_dummies(training_df[column], prefix=column)
        
        columnnames = list(dummies.columns.values)
       
        dummies = dummies.iloc[:,:-1]
        
        training_df = pd.concat([training_df, dummies], axis=1)
        training_df = training_df.drop([column], axis=1)

        
        unique_list = validation_df[column].unique()
        
        dummies = pd.get_dummies(validation_df[column], prefix=column)
        columnnames = list(dummies.columns.values)
        dummies = dummies.iloc[:,:-1]
        validation_df = pd.concat([validation_df, dummies], axis=1)
        validation_df = validation_df.drop([column], axis=1)

        unique_list = test_df[column].unique()
        dummies = pd.get_dummies(test_df[column], prefix=column)
        columnnames = list(dummies.columns.values)
        dummies = dummies.iloc[:,:-1]
        test_df = pd.concat([test_df, dummies], axis=1)
        test_df = test_df.drop([column], axis=1)

columnnames = list(training_df.columns.values)

neural_network = [[],[]]

#assigning random weights to hidden layer
for i in range(number_neurons):
    neuron=[]
    for j in range(len(columnnames)):
        new_weight = random.uniform(-0.1, 0.1)
        neuron.append(new_weight)
    neural_network[0].append(neuron)

neuron=[]

#assigning random weights to output layer
for j in range(number_neurons+1):
    new_weight = random.uniform(-0.1, 0.1)
    neuron.append(new_weight)
neural_network[1].append(neuron)

# calculate out k function
def calculate_outk(neuron,row):
    net = neuron[0]
    for i in range(1, len(neuron)):
        net = net + (row[i]*neuron[i])
    out = 1/(1+math.exp(-net))
    return out

# create 2d confusion matrix (2d array)
all_unique = dataframe[dataframe.columns[0]].unique()
twolist = []
place_dict = {}

for x in range(len(all_unique)):
    newlist=[]
    for y in range(len(all_unique)):
        newlist.append(0)
    twolist.append(newlist)

for x in range(len(all_unique)):
    place_dict[all_unique[x]] = x

# calculate out o function
def calculate_outo(neuron,activation):
    net=neuron[0]
    for i in range(1,len(neuron)):
        net=net+(activation[i-1]*neuron[i])
    out=1/(1+math.exp(-net))
    return out

activation=[0]*number_neurons
activation1=[0]*number_neurons
feedback=[0]*number_neurons

#neural network
accuracy=0
epoch=0

while epoch<500 and accuracy<0.99:
    
    true=0

    #each instance 
    for i in range(len(training_df)):
        
        row=training_df.iloc[i].to_numpy()
        
        #calculating out_k for each neuron in the hidden layer
        for neuron in range(len(neural_network[0])):
            
            activation[neuron]=calculate_outk(neural_network[0][neuron],row)
        
        #calculating out_o for the output neuron
        out_o=calculate_outo(neural_network[1][0],activation)

        #calculating the error of the neural network's prediction
        error=row[0]-out_o

        #calculating feedbacks for the neurons to understand their responsibility in error

        feedback_output=out_o*(1-out_o)*error

        for neuron in range(len(neural_network[0])):
            feedback[neuron]=activation[neuron] * (1-activation[neuron])* neural_network[1][0][neuron+1]*feedback_output
        
        #update weights for output neuron
        neural_network[1][0][0]=neural_network[1][0][0]-learning_rate*(-1*feedback_output)

        for neuron in range(1,len(neural_network[1][0])):
            neural_network[1][0][neuron]=neural_network[1][0][neuron]-learning_rate*(-activation[neuron-1]*feedback_output)

        #update weights for each neuron k
        
        for neuron in range(len(neural_network[0])):
            neural_network[0][neuron][0]=neural_network[0][neuron][0]-learning_rate*(-1*feedback[neuron])
            for weight in range(1,len(neural_network[0][neuron])):
                neural_network[0][neuron][weight]=neural_network[0][neuron][weight]-learning_rate*(-row[weight]*feedback[neuron])

    for i in range(len(validation_df)):

        row=validation_df.iloc[i].to_numpy()

        #calculating out_k for each neuron in the hidden layer
        for neuron in range(len(neural_network[0])):
            activation1[neuron]=calculate_outk(neural_network[0][neuron],row)

        #calculating out_o for the output neuron
        out_o=calculate_outo(neural_network[1][0],activation1)
        if out_o >= threshold:
            predicted=1
        else:
            predicted=0
        if predicted==row[0]:
            true+=1

    accuracy = true/len(validation_df)
    epoch+=1

for i in range(len(test_df)):
    row=test_df.iloc[i].to_numpy()

    #calculating out_k for each neuron in the hidden layer
    for neuron in range(len(neural_network[0])):
        activation1[neuron]=calculate_outk(neural_network[0][neuron],row)

    #calculating out_o for the output neuron
    out_o=calculate_outo(neural_network[1][0],activation1)
    if out_o >= threshold:
        predicted=1
    else:
        predicted=0

    # increment confusion matrix
    column = place_dict[predicted]
    row = place_dict[row[0]]
    twolist[row][column]+=1
        
# create file name
length = len(dataset)
abrev = dataset[0:length-4]
name = "results_" + abrev + "_" + str(number_neurons) + "n_" + str(learning_rate) + "r_" + str(threshold) + "t_" + str(training_percentage) + "p_" + str(random_seed) + ".csv"

final_labels = all_unique
final_labels = final_labels.tolist()
final_labels.append("")

# create new csv file
with open(name, 'w', newline='') as newfile:
    # initialize csv
    write = csv.writer(newfile)
    write.writerow(final_labels)
    count=0
    # write each row to csv
    for row in twolist:
        row.append(all_unique[count])
        write.writerow(row)
        count+=1       