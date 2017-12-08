from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
#xs=np.array(range(1,7),dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
	m = ( ((mean(xs)* mean(ys)) - mean(xs * ys))/
	      ((mean(xs)**2) - mean(xs**2)) )
	b = mean(ys) - m * mean(xs)
	return m,b

#ys_orig - ys of your best fit line
#ys_line - ys of your points of dataset

def squared_error(ys_original,ys_line):
	return sum((ys_line - ys_original)**2)

	
def create_dataset(how_much,variance,step=2,correlation=False):
	ys=[]
	val = 1
	for i in range(how_much):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation=="pos":
			val += step
		elif correlation and correlation=="neg":
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)
					
	
def coefficient_of_determination(ys_reggr,ys):
	ys_mean = [mean(ys) for y in ys] # mean of all y's
	squared_error_y_hat = squared_error(ys,ys_reggr) #inp line
	squared_error_y_mean = squared_error(ys,ys_mean) #mean of datasets
	return 1- (squared_error_y_hat)/(squared_error_y_mean)
	
xs,ys = create_dataset(100,70,2,"pos")
print "ys",ys
m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs] #y values
predict_x = 8
predict_y = (m* predict_x) + b

r_sqd = coefficient_of_determination(regression_line,ys)#to calculate accuracy
print "Accuracy",r_sqd
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s=100,color='g')
plt.plot(xs,regression_line)
plt.show()

#r squred(co-efficient of determnation) - to determne accuracy of the model
#done using squared error - original y's distance from the actual y of the best-fit line
#r-squared should be high!i

