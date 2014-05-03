# Copyright (C) 2014  Caleb Lo
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Machine Learning
# Programming Exercise 1: Linear Regression
# Problem: Predict profits for a food truck given data for profits/populations of various cities
# Linear regression with one variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class InsufficientIterations(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class InsufficientTrainingExamples(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Plot data
def plotData(x,y):
    "Plot data"
    plt.plot(x,y,'rx',markersize=18)
    plt.ylabel('Profit in $10,0000s',fontsize=18)
    plt.xlabel('Population of City in 10,000s',fontsize=18)

    return None

# Compute cost function J(\theta)
def computeCost(X,y,theta):
    "Compute cost function J(\theta)"
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    diffVec = np.subtract(np.dot(X,theta),y)
    diffVecSq = np.multiply(diffVec,diffVec)
    jTheta = (np.sum(diffVecSq,axis=0))/(2*numTrainEx)

    return(np.asscalar(jTheta))

# Run gradient descent
def gradientDescent(X,y,theta,alpha,numiters):
    "Run gradient descent"
    if (numiters <= 0):
        raise InsufficientIterations('numiters <= 0')
    numTrainEx = y.shape[0]
    if (numTrainEx == 0):
        raise InsufficientTrainingExamples('numTrainEx = 0')
    jThetaArray = np.zeros((numiters,1))
    for thetaIndex in range(0,numiters):
        diffVec = np.subtract(np.dot(X,theta),y)
        diffVecTimesX = np.c_[np.multiply(diffVec,np.reshape(X[:,0],(numTrainEx,1))),np.multiply(diffVec,np.reshape(X[:,1],(numTrainEx,1)))]
        thetaNew = np.subtract(theta,alpha*(1/numTrainEx)*np.reshape(np.sum(diffVecTimesX,axis=0),(2,1)))
        jThetaArray[thetaIndex] = computeCost(X,y,thetaNew)
        theta = thetaNew

    return(theta)

# Main function
def main():
    "Main function"
    print("Plotting Data ...")
    foodTruckData = np.genfromtxt("../foodTruckData.txt",delimiter=",")
    returnCode = plotData(foodTruckData[:,0],foodTruckData[:,1])
    plt.show()
    input("Program paused. Press enter to continue.")
    print("")
    print("Running Gradient Descent ...")
    numTrainEx = foodTruckData.shape[0]
    onesVec = np.ones((numTrainEx,1))
    xMat = np.c_[onesVec,foodTruckData[:,0]]
    thetaVec = np.zeros((2,1))
    yVec = np.reshape(foodTruckData[:,1],(numTrainEx,1))

    # Compute initial cost
    initCost = computeCost(xMat,yVec,thetaVec)
    print("ans = %.3f" % initCost)

    # Run gradient descent
    iterations = 1500
    alpha = 0.01
    thetaFinal = gradientDescent(xMat,yVec,thetaVec,alpha,iterations)
    print("Theta found by gradient descent: %s" % np.array_str(np.transpose(np.round(thetaFinal,6))))
    returnCode = plotData(foodTruckData[:,0],foodTruckData[:,1])
    plt.hold(True)
    plt.plot(xMat[:,1],np.dot(xMat,thetaFinal),'b-',markersize=18)
    plt.legend(('Training data','Linear regression'),loc='lower right')
    plt.hold(False)
    plt.show()

    # Predict profit for population size of 35000
    predProfit1 = np.asscalar(np.dot(np.array([1,3.5]),thetaFinal))
    predProfit1Scaled = 10000*predProfit1
    print("For population = 35,000, we predict a profit of %.6f" % predProfit1Scaled)

    # Predict profit for population size of 70000
    predProfit2 = np.asscalar(np.dot(np.array([1,7.0]),thetaFinal))
    predProfit2Scaled = 10000*predProfit2
    print("For population = 70,000, we predict a profit of %.6f" % predProfit2Scaled)
    input("Program paused. Press enter to continue.")
    print("")
    print("Visualizing J(theta_0, theta_1) ...\n")
    theta0Vals = np.linspace(-10,10,num=100)
    theta1Vals = np.linspace(-1,4,num=100)
    jVals = np.zeros((theta0Vals.shape[0],theta1Vals.shape[0]))
    for theta0Index in range(0,theta0Vals.shape[0]):
        for theta1Index in range(0,theta1Vals.shape[0]):
            tVec = np.vstack((theta0Vals[theta0Index],theta1Vals[theta1Index]))
            jVals[theta0Index,theta1Index] = computeCost(xMat,yVec,tVec);
    jValsTrans = np.transpose(jVals)
    theta0ValsX,theta1ValsY = np.meshgrid(theta0Vals,theta1Vals)
    jValsReshape = jValsTrans.reshape(theta0ValsX.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(theta0ValsX,theta1ValsY,jValsReshape,rstride=1,cstride=1,cmap=cm.jet)
    ax.set_ylabel(r'$\Theta_1$',fontsize=18)
    ax.set_xlabel(r'$\Theta_0$',fontsize=18)
    plt.show()
    plt.contour(theta0ValsX,theta1ValsY,jValsReshape,levels=np.logspace(-2,3,20))
    plt.ylabel(r'$\Theta_1$',fontsize=18)
    plt.xlabel(r'$\Theta_0$',fontsize=18)
    plt.hold(True)
    plt.plot(thetaFinal[0],thetaFinal[1],'rx',markersize=18,markeredgewidth=3)
    plt.hold(False)
    plt.show()

# Call main function
if __name__ == "__main__":
    main()
