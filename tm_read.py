import re
import pickle
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import os
from os.path import join
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline


basepath = "F:/Text Mining/Assignment/data/"
narratives=[]

def readfile(file):
    f=open(file, "r")
    if f.mode == 'r':
        contents =f.read()
        #print(body[1])
        #return body[1]
        return contents
    
def para_div(content):
    paras = content.split("Report Narratives")
    narratives.append(paras)
    for i in narratives[1:2]:   
        a=preprocess(str(i))
        vectorizer = TfidfVectorizer(max_df=0.7, max_features=2500,
                             min_df=3,
                             use_idf=True)
        X = vectorizer.fit_transform(a)
        X.shape


    #print('paras',paras[0])
    para = paras[1].split("Time / Day Date :")
    return para
def preprocess(i):
    tokens = nltk.word_tokenize(i)
    #print(tokens)          
    unique = set(tokens)

    #print(unique)
    # What about bigrams and trigrams?
    #every three continues token along the way
    #preceding word and following word info
    bigr = nltk.bigrams(tokens[:10])
    trigr = nltk.trigrams(tokens[:10])
    tokens[:10]
    #print(list(bigr))
    list(trigr)

    # Back to text preprocessing: remove punctuations
    tokens_nop = [ t for t in tokens if t not in string.punctuation ]
    #print(tokens[:50])
    #print(tokens_nop[:50])
    len(set(tokens_nop))

    # Convert all characters to Lower case
    tokens_lower=[ t.lower() for t in tokens_nop]
    #print(tokens_lower[:50])
    
    len(set(tokens_lower))
    stop = stopwords.words('english')+['â€', '”','...', "..."]
    # Remove all these stopwords from the text
    tokens_nostop=[ t for t in tokens_lower if t not in stop ]
    #print(tokens_nostop[:50])
    len(tokens_lower)
    len(tokens_nostop)
    FreqDist(tokens_nostop).most_common(50)
    snowball = nltk.SnowballStemmer('english')
    tokens_snow = [ snowball.stem(t) for t in tokens_nostop ]
    #print(tokens_snow[:50])
    len(set(tokens_snow))
    
    return(tokens_snow)
    

def sep_data(paras):
    incident_description = ""
    try:
        desc = paras.split("Narrative")
        #print(desc[0])
        for i in range(1,len(desc)):
            incident_description = incident_description + desc[i]
            #print('incident_description',incident_description.count("incident_description"))
        #print("Incident Description : \n"+incident_description+"\n")
        #just get desc[1:] gives the narratives and synopsis
        #dont forget to add :,narrative and synopsis into stop word
    except:
        desc[0] = paras

    try:   
        assessments = desc[0].split("Assessments")
        #print("Assessment :\n"+assessments[1]+"\n")
    except:
        assessments[0] = desc[0]

    try:
        events = assessments[0].split("Events")
        #print("Events :\n"+events[1]+"\n")
        
    except:
        events[0] = assessments[0]

    person_involved = ""
    try:
        person = events[0].split("Person")
        for i in range(1,len(person)):
            person_involved = person_involved + person[i]
        #print("People involved :\n"+ person_involved +"\n")
    except:
        person[0] = events[0]

    component_involved = ""
    try:
        comp = person[0].split("Component")
        for i in range(1,len(comp)):
            component_involved = component_involved + comp[i]
        #print("Components: \n"+ component_involved +"\n")
    except:
        comp[0] = person[0]

    try:
        aircraft = comp[0].split("Aircraft")
        #print("Aircraft: \n"+aircraft[1]+"\n")
    except:
        aircraft[0] = comp[0]

    try:
        environment = aircraft[0].split("Environment")
        #print("Environment :\n"+environment[1]+"\n")
    except:
        environment[0] = aircraft[0]

    try:
        place = environment[0].split("Place")
        #print("Place :\n"+place[1]+"\n")
        #print("Time: \n"+ place[0]+"\n")
        #place[0] holds time and date
    except:
        place[1] = ""


for (dirname, dirs, files) in os.walk(basepath):
   for filename in files:
       if filename.endswith('.txt') :
            thefile = os.path.join(dirname,filename)
            #print(thefile)      
            data = readfile(thefile)
            paras= para_div(data)
            for i in range(1,len(paras)):
                #print("\n\n Incident " + str(i) + "\n")
                sep_data(paras[i])

