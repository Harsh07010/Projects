import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("data.csv",error_bad_lines=False)
# print(data.head())

data=data.dropna()
data["strength"]=data["strength"].map({0:"Weak",
                                       1:"Medium",
                                       2:"Strong"})

#print(data.sample(5))

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character

x=np.array(data["password"])
y=np.array(data["strength"])

tdif=TfidfTransformer(tokenizer=word)
x=tdif.fit_transform(x)
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.05,random_state=42)


model=RandomForestClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

import getpass
user=getpass.getpass("Enter password")
data=tdif.transform([user]).toarray()
output=model.predict(data)
print(output)
