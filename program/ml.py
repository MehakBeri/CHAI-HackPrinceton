#ml model which provides parameters to the different variables

# Class MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as activation function. Therefore, it uses the square error as the loss function, and the output is a set of continuous values.
#
# MLPRegressor also supports multi-output regression, in which a sample can have more than one target.

import nlp as nlpFile
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

if __name__=="__main__":
    print("========================================================================")
    trainingData="G:/hackPrinceton/data/chai/training/7.xlsx"
    data=nlpFile.getFeatures(trainingData,"Laughter is the shortest distance between two people")
    data_narray=numpy.array(data)
    featureList=[]
    featureList.append(data[0][0])
    # featureList.append(data[:][3])
    # featureList.append(data[:][4])
    # featureList.append(data[:][5])
    # featureList.append(data[:][6])
    # featureList.append(data[:][7])
    # featureList.append(data[:][8])
    print(featureList)
    # pipeline = Pipeline([
    #     ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
    #     ('neural network', Classifier(layers=[Layer("Softmax")], n_iter=25))])
    # pipeline.fit(X_train, y_train)
