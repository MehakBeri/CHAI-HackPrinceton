#ml model which provides parameters to the different variables

# Class MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as activation function. Therefore, it uses the square error as the loss function, and the output is a set of continuous values.
#
# MLPRegressor also supports multi-output regression, in which a sample can have more than one target.
import nlp as nlpFile
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

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

    names=['id','number of words','coherence','grammar','similarity']
    bos=pd.DataFrame(j.get('0'))
    bos.columns=names
    bos.head()
    bos['score']=j.get('1')
    x=bos.drop('score',axis=1)
    lm=sklearn.linear_model.LinearRegression()
    lm.fit(x,bos.score)
    print('number of coefficients:',len(lm.coef_))

    test = "G:/hackPrinceton/CHAI/data/chai/training/7.xlsx"
    testData = nlpFile.getFeatures(test, "Laughter")
    t = refineData(testData)
    names = ['id', 'number of words', 'coherence', 'grammar', 'similarity']
    bos1 = pd.DataFrame(t.get('0'))
    bos1.columns = names
    bos1.head()
    bos1['score'] = t.get('1')
    x1 = bos1.drop('score', axis=1)
    print(lm.predict(x1))
    mseFull=np.mean((bos1.score- lm.predict(x1))**2)
    print(mseFull)