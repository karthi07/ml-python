#Classification on Iris data set
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
feature_name = data['feature_names']
target = data['target']
labels = data['target_names'][target]

#print labels[:10]
#print features.shape[1]

# Plots

import matplotlib.pyplot as plt

for t,marker,c in zip(xrange(3),">ox","rgb"):
  plt.scatter(features[target==t,3],features[target==t,2],marker=marker,c=c)

#plt.show()



# First Classifier

plength = features[:,2]
is_setosa = (labels == "setosa")

#calculating plen max and min
setosa_max = plength[is_setosa].max()
non_setosa_min = plength[~is_setosa].min()

#print "Maximum plength of setsa : {0}.".format(setosa_max)

#print "Minimum plength of non setsa : {0}.".format(non_setosa_min)
# max setosa is 1.9 and min of nonsetosa is 3 
# so if <2 it is setosa
# simple classifier 


plength = features[:,2]

#print plength[:5] < 2
#if plength[0] < 2:
#  print "Setosa"
#else :
#  print "Virginica or Versicolor"


#  More Complex



#split data

test = np.tile([True,False],50)
train = ~test


features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')



def learn_model(features,label):
  best_acc = -1.0
  for fi in xrange(features.shape[1]):
    thresh = features[:,fi].copy()
    thresh.sort()
    for t in thresh:
      pred = (features[:,fi] > t)
      acc = (pred == label).mean()	
      if acc > best_acc:
        best_acc = acc
        best_fi = fi
        best_t = t
  return best_fi,best_t

def apply_model(features,model):
  fi,t = model
  return features[:,fi] > t

def accuracy(features,model,labels):
  res = apply_model(features,model)
  return (res == labels).mean()



model = learn_model(features[train],virginica[train])
print ('Training accuracy {0:.1%}.'.format(accuracy(features[train],model,virginica[train])))
print ('Testing accuracy {0:.1%}'.format(accuracy(features[test],model,virginica[test])))


#Load wheat seed dataset

data = []
label = []

with open("seeds.tsv") as dfile:
  for line in dfile:
    token = line.strip().split("\t")
    data.append( [float(tk) for tk in token[:-1]])
    label.append(token[-1])
  data = np.array(data)
  label = np.array(label)

print data.shape


#knn classifer
from knn import *

features = data


#cross validate


def cross_validate(features,labels):
    error = 0.0
    for fold in range(10):
        train = np.ones(len(features),bool)
        train[fold::10]=0
        test=~train
        model = learn_model(1,features[train],labels[train])
        test_error = accuracy(features[test],labels[test],model)
        error += test_error

    return error/ 10.0

#error = cross_validate(features,label)

#print ("The ten fold cross validate error is {0:.1%}".format(error))

features -= features.mean(0)
features /= features.std(0)
error = cross_validate(features, label)
print('Ten fold cross-validated error after z-scoring was {0:.1%}.'.format(error))




