#ml model which provides parameters to the different variables

# Class MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as activation function. Therefore, it uses the square error as the loss function, and the output is a set of continuous values.
#
# MLPRegressor also supports multi-output regression, in which a sample can have more than one target.

import nlp as nlpFile
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import StandardScaler

def refineData(data):
    essay_id = np.asarray([d[0] for d in data])
    score = np.asarray([d[3] for d in data])
    N_words = np.asarray([d[4] for d in data])
    Coherence = np.asarray([d[5] for d in data])
    NGram = np.asarray([d[6] for d in data])
    GrammarCheck = np.asarray([d[7] for d in data])
    Similarity = np.asarray([d[8] for d in data])
    xtrain = []
    ytrain = []
    for i in range(len(score)):
        temp = []
        temp.append(essay_id[i])
        temp.append(N_words[i])
        temp.append(Coherence[i])
        temp.append(GrammarCheck[i])
        temp.append(Similarity[i])
        xtrain.append(temp)
        ytrain.append(score[i])
    return {'0':np.array(xtrain),'1':np.array(ytrain)}

def calculateAccuracy(actual,calculated):
    score=0
    for i in range(len(actual)):
        if ((actual[i]<=calculated[i]+5) and (actual[i]>=calculated[i]-5)) or ((actual[i]<=calculated[i]+5) and (actual[i]>=-calculated[i]+5)) :
            score=score+1
    percent= score/len(actual)
    print("Accuracy is: ",percent)

if __name__=="__main__":
    print("========================================================================")
    trainingData="G:/hackPrinceton/CHAI/data/chai/training/7.xlsx"
    data=nlpFile.getFeatures(trainingData,"Laughter")
    j=refineData(data)
    # print(j.xtrain)
    # print(j.ytrain)
    pipeline = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', Classifier(layers=[Layer("Softmax")], n_iter=25))])
    pipeline.fit(np.asarray(j.get('0')), np.asarray(j.get('1')))
    nn = Classifier(
        layers=[
            Layer("Maxout", units=100, pieces=2),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=25)
    nn.fit(j.get('0'), j.get('1'))
    test="G:/hackPrinceton/CHAI/data/chai/training/7.xlsx"
    testData=nlpFile.getFeatures(test,"Laughter")
    t=refineData(testData)
    y_actual= t.get('1')
    x_test=t.get('0')
    y_calculated = nn.predict(x_test)
    # print(y_actual)
    # print(y_calculated)
    calculateAccuracy(y_actual,y_calculated)