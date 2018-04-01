import xlrd
from sklearn.feature_extraction.text import CountVectorizer
import language_check
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

def getData(address):
    workbook = xlrd.open_workbook(address, "rb")
    sheets = workbook.sheet_names()
    data = []
    for sheet_name in sheets:
        sh = workbook.sheet_by_name(sheet_name)
        for rownum in range(5): #sh.nrows
            row_vals = sh.row_values(rownum)
            temp=[]
            temp.append(row_vals[0])
            temp.append(row_vals[1])
            temp.append(row_vals[2])
            temp.append(row_vals[5])
            data.append(temp)
    return data

def getNumbers(data):
        for d in data:
            d.append(len(d[2]))
        return data

def BagOfWords(data):
    corpus = []
    for dataRow in data:
        corpus.append(dataRow[2])

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    ar=vectorizer.fit_transform(corpus).todense().tolist()
    i=0
    for x in data:
        x.append(ar[i])
        i=i+1
    return data

def grammar(data):
    tool = language_check.LanguageTool('en-US')
    for d in data:
        text=d[2]
        matches = tool.check(text)
        d.append(len(matches))
    return data

def stopWords(dataSet):
    stop_words = set(stopwords.words('english'))
    for d in dataSet:
        data=d[2]
        word_tokens = word_tokenize(data)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        nonGibberish = len(filtered_sentence)/len(word_tokens)
        d.append(nonGibberish)
    return dataSet

def similarity(dataSet,theme):
    stop_words = set(stopwords.words('english'))
    themeSynsets= wn.synsets(theme, pos=wn.NOUN)
    for d in dataSet:
        data=d[2]
        word_tokens = word_tokenize(data)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        #check similarity
        s=0.0
        for word in filtered_sentence:
            w_synsets= wn.synsets(word)
            for w_s in w_synsets:
              for syns in themeSynsets:
                    if(w_s.path_similarity(syns)):
                      s = s + w_s.path_similarity(syns)
        d.append(s)
    return dataSet

def getFeatures(address,theme):
    d = getData(address)
    data1 = getNumbers(d)
    data1_2 = stopWords(data1)
    data2 = BagOfWords(data1_2)
    data3 = grammar(data2)
    data4 = similarity(data3, theme)
    return data4

if __name__=="__main__":
    print("========================================================================")
    trainingData="G:/hackPrinceton/data/chai/training/7.xlsx"
    #GRADING CRITERIA:
    # >90: A; 80-89:A- ; 70-79:B; 60-69:B- ; 50-59:C; 40-49:C- ; <30: F
    # test Data has essay id, essay set, essay, score. need to add the following features to each data tuple:
    d = getData(trainingData)
    # number of words
    data1 = getNumbers(d)
    # semantic coherence - remove gibberish by removing all stop words- calculate percentage imp words out of total words
    data1_2=stopWords(data1)
    # bag of words/n-gram
    data2 = BagOfWords(data1_2)
    # grammar - use language_check library- the lesser number of matches, the more perfect grammar has  been used.
    #so this parameter should have a negative weight assigned to it
    data3= grammar(data2)
    # clustering using latent semantic analysis
    # plagiarism
    # similarity measure - similarity between prompt words and essay non-stop words-using princeton's wordnet
    #For dataset 7- the theme is "patience"
    data4=similarity(data3,"Laughter is the shortest distance between two people")
    print(data4[0])

# DATA AT THIS TIME HAS THE FOLLOWING PARAMETERS:
# 0: essay id
# 1: essay set
# 2: essay
# 3: score
# 4: number of words
# 5: semantic coherence %age
# 6: n-gram
# 7: grammar check parameter
# 8: similarity index






