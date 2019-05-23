import numpy as np
import bokeh 
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.core.properties import value
import pandas as pd
from bokeh.io import show, output_notebook
from sklearn import preprocessing
# output_notebook()
dwh_train = pd.read_csv('DWH_Training.csv',header=None)
big_dwh_train = pd.read_csv('BIG_DWH_Training.csv',header = None)
dwh_test = pd.read_csv('DWH_test.csv',header= None)


def initialise_weight_and_bias_from_NCC():
    return [-11.079205,-16.611210], 2976.8103885587443
#soft margin Linear SVM
def update_params_sgd(learn_rate, w, b, features, labels, c):
    dw_subgradient= 0
    db_subgradient = 0
    for i in range(len(features)):
        #y(w.T * xi)
        if labels[i] * (np.dot(np.transpose(w),features[i]) + b) < 1:
            #y*xi
            dw_subgradient += (labels[i] * features[i])
            #yi
            db_subgradient += (labels[i])
        else:
            dw_subgradient += 0
            db_subgradient += 0
    w = w - learn_rate * (w - c * dw_subgradient)
    b = b - learn_rate * (- c * db_subgradient)
    return w,b
    
def splitData(data, fold_num,fold_size,n_folds):
    curr_index = fold_num * fold_size
    if fold_num == n_folds-1:
        train = data[0:curr_index]
        test = data[curr_index : ]
    else:
        train = data[0:curr_index].append(data[curr_index + fold_size:])
        test = data[curr_index : curr_index + fold_size]
    return train, test

def calc_accuracy(train, test, t , c, batch ):
    w,b = svm(np.array(train.iloc[:,1:3]), np.array(train.iloc[:,3]),t, c, batch)
    correct_results = 0
    test_x = np.array(test.iloc[:,1:3])#test features
    test_y = np.array(test.iloc[:,3])#test labels
    for i in range(test.shape[0]):
        pred = np.dot(np.transpose(w),test_x[i]) + b
        if pred > 0 and test_y[i] == 1:
            correct_results +=1
        if pred < 0 and test_y[i] == -1:
            correct_results += 1
    return correct_results / len(test_x),w,b

def svm(features,labels,num_iterations, c, batch_size):
    w,b = initialise_weight_and_bias_from_NCC()
    learn_rate = 1/num_iterations
    for i in range(1, num_iterations+1):
        learn_rate =1  / i
        data_indexes = np.random.choice(np.arange(0,features.shape[0]),size=batch_size)
        w,b = update_params_sgd(learn_rate,np.array(w), b, features[data_indexes],labels[data_indexes], c)
    return w,b

#t-times k-fold cross validation   
def cross_validate_sgd(k,data,num_iterations,c,batch):
    fold_size = len(data) // k
    acc=[]
    w=0
    b=0
    #assume t to be 1
    for fold_num in range(k):
        train, test = splitData(data, fold_num, fold_size,k)
        accuracy,w,b = calc_accuracy(train,test,num_iterations,c,batch)
        acc.append(accuracy)
    return np.sum(acc)/ len(acc),w,b


#####3b
C = [.1,1,10]
B = [1,10,50]
final_acc = final_w = final_b = final_batch = final_C=0
for c in C:
    for batch in B:
        accuracy,w,b = cross_validate_sgd(10,dwh_train,10000,c,batch)
        if accuracy > final_acc:
            final_acc = accuracy
            final_w = w
            final_b = b
            final_C = c
            final_batch = batch
        print("Accuracy for c ",c," and b ",batch," ", accuracy )

print('Final Accuracy',final_acc)
print('Final Weight', final_w)
print('Final Bias',final_b)
print('Final C', final_C)
print('Final Batch', final_batch)
            
            
#####3c
#c) Scatter Plot
#Using w = [-16.97434913  -1.22097908] ; b = 2981.992588038288 from previous result
def predictValue(x1):
    return ((-final_w[0] / final_w[1]) *  x1) + (- final_b / final_w[1])

dwh_train_2 = pd.read_csv('DWH_Training.csv',names = ['Row_Number', 'Height', 'Weight', 'Gender'])
df_male = dwh_train_2.groupby('Gender').get_group(1)
df_female = dwh_train_2.groupby('Gender').get_group(-1)

p = figure(title = " Disney Demography Institute Survey", plot_width=800, x_axis_label='Height in cms',
           y_axis_label = 'Weight in kgs' )
p.circle(x = df_male.Height,y = df_male.Weight, size=10, color="blue",alpha=0.4,legend="Male",name="Male")
p.circle(x = df_female.Height, y = df_female.Weight, size=10, color="pink",alpha=0.4,legend="Female",name="Female")

xmin = min(dwh_train_2.Height)
xmax = max(dwh_train_2.Height)
p.line(x=[xmin,xmax],y=[predictValue(xmin),predictValue(xmax)],color='black')

show(p)



#####3d Compare accuracy with Liblinear
# Using C = 1 batch_size = 50
correct_results = 0
test_x = np.array(dwh_test.iloc[:,1:3])#test features
test_y = np.array(dwh_test.iloc[:,3])#test labels
for i in range(dwh_test.shape[0]):
    pred = np.dot(np.transpose(final_w),test_x[i]) + final_b
    if pred > 0 and test_y[i] == 1:
        correct_results +=1
    if pred < 0 and test_y[i] == -1:
        correct_results += 1
print("Accuracy using SVM - SGD ",correct_results / len(test_x) * 100,"%")
print("Accuracy using Liblinear 90.90%")


#####3e
import time

def dispTimeInMinutes(val):
    hours , rem = divmod(val, 3600)
    minutes, seconds = divmod(rem,60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

ticks = time.time()
big_features = np.array(big_dwh_train.iloc[:,1:3])
big_labels = np.array(big_dwh_train.iloc[:,3])
start_time = time.time()
for i in range(10):
    print('Run #',i)
    w,b = svm(big_features,big_labels,10,1,20000)
end_time = time.time()
print("SVM(via SGD) execution time :", 
                    dispTimeInMinutes(end_time - start_time))


#LIBLINEAR
y = (big_dwh_train.iloc[:,-1]).values
x = (big_dwh_train.iloc[:,[1,2]]).values
#SCALING
x_scaled=preprocessing.scale(x)
start = time.time()
for i in range(10):
    model=train(y,x_scaled,'-s 3 -B 8.5')
    p_label, p_acc, p_val = predict(y, x_scaled, model)
stop = time.time()

print("Liblinear execution time :", dispTimeInMinutes(stop - start))