

import numpy # numpy library is used for works with numpy array
import matplotlib.pyplot as plt # matplotlib is used for visualziatioon purpose 
import seaborn as sns # and seaborn is built on the of the matplotlib to plot the graph
import random # random is used to generate random data 


def studentReg(ages_train , net_worths_train):
  from sklearn.linear_model import LinearRegression
  reg = LinearRegression().fit(ages_train, net_worths_train)
  return reg

  numpy.random.seed(42) # seed() fuction is used to generate same random number agian and again enen you perform multiple time it's gives the same valuewe
ages = [] 
for ii in range(250):   # this loops genrate 250 random number using between the value of (18 to  75)
    ages.append(random.randint(18,75))
net_worths = [ii*6.25 + numpy.random.normal(scale= 40)for ii in ages]
ages = numpy.reshape(numpy.array(ages),(len(ages),1))
net_worths = numpy.reshape(numpy.array(net_worths),(len(net_worths),1))

from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages,net_worths)
# by passing the ages and net_worths to the taining  test day it split data into two set's 
reg1 = studentReg(ages_train,net_worths_train)
print("coffecient",reg1.coef_)
print("intercept",reg1.intercept_)


print("Training data",reg1.score(ages_train,net_worths_train))
print("Testing data",reg1.score(ages_test,net_worths_test))

# ploting of data
plt.figure(figsize=(12,10))
sns.regplot(x = ages_train, y=net_worths_train,scatter=True,color="b",marker="*")
plt.xlabel("Ages Train")
plt.ylabel("Net Worth Train")
plt.title("Regression plot")
plt.figure(figsize=(12,10))
plt.scatter(ages_train,net_worths_train,color="b",label="train data",marker="*")
plt.scatter(ages_test,net_worths_test,color='r', label="test data")
plt.plot(ages_test,reg1.predict(ages_test))
plt.xlabel(ages)
plt.ylabel("Net Worth")
plt.legend(loc=2)
plt.show()
