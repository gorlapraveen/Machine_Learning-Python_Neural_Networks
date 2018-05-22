#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
##########################################################
##For calculation off nuerons output we use, weights W(i).ip(i)=w1.ip1+w2.ip2+w3.ip3;
##we use sigmod function 1/1=exp(-w(i).ip(i)) function for output calculation 
#and later on for optimizing weights by calculating the error between the training output and th neuronal network output, based on this we adjust this weights, we do this for 100000 iteration sof continusly updating the weights so that we achiceve the optimal output.

###Adjust weight by =error.input.SigmodCurveGradient(output)=error.input.output(1-output)

## now we select the random weights

from numpy import exp, array, random, dot
training_inputs=array([[4,0,2],[3,0,0],[9,4,0],[12,0,10]])
training_outputs=array([[4,3,9,12]]).T
random.seed(1); ##You are seeding the random number so that everytime it start generating the same numbers
sample_weights=40 * random.random((3,1)) - 20#3 ip connectio , 1 output, 3x1 matrix with values range -1 to1, actually random.random() generates from 0 to 1 with mean 0.5, so i order to make the mean as 0 weprint 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))) use 2*random.random()-1 ranging from -1 to 1 with mean 0;
for iteration in xrange(10000):
    outputs=1/(1+exp(-(dot(training_inputs,sample_weights)))) # nueral o/p cal using inputs sigmod function
    sample_weights+=dot(training_inputs.T,(training_outputs-outputs)*outputs*(1-outputs))
print 1/(1 + exp(-(dot(array([9,10,10]), sample_weights))))

