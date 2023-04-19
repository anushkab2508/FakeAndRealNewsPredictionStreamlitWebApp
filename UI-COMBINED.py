from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import time as ts
from datetime import timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import time
import string
data_fake=pd.read_csv("Fake.csv")
data_true=pd.read_csv("True.csv")

#data_fake.head()

#data_true.head()

data_fake["class"]=0
data_true["class"]=1

#to identify which one tis true(1) and which one is false(0)"""

#data_fake.shape,data_true.shape 
#shape return the no of columns,rows of dataframe

#the shape method return shape of array ,length of corresponding array gives rows and columns

data_fake_manual_testing=data_fake.tail(10)
for i in range(23480,23470,-1):
  data_fake.drop([i], axis = 0, inplace =True)

data_true_manual_testing=data_true.tail(10)
for i in range(21416,21406,-1):
  data_true.drop([i], axis = 0, inplace =True)

#data_fake.shape, data_true.shape

data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1

#data_fake_manual_testing.head(10)

#data_true_manual_testing.head(10)

data_merge = pd.concat([data_fake,data_true],axis=0)
data_merge.head(10)

#data_merge.columns  (showing the merged columns)

data =data_merge.drop(['title','subject','date'],axis=1)

data.isnull().sum()

data=data.sample(frac=1)

#data.head()

data.reset_index(inplace = True)
data.drop(['index'],axis = 1,inplace=True)

#data.columns

#data.head()

def wordopt(text):
  text=text.lower()
  text=re.sub('\[.*?\]','',text)
  text = re.sub("\\W"," ", text)
  text = re.sub('https?://\S+|www\.\S+','',text)
  text= re.sub('<.*?>+','',text)
  text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
  text = re.sub('\n','',text)
  text= re.sub('\w*\d\w*','',text)
  return text

#to remove these characters from dataset"""

data['text'] = data['text'].apply(wordopt)

x= data['text']
y=data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# from sklearn.linear_model import LogisticRegression 
# LR= LogisticRegression()
# LR.fit(xv_train, y_train)

# pred_lr=LR.predict(xv_test)

# LR.score(xv_test,y_test)
def output_label(n):
  if n==0:
    return "fake news"
  elif n==1:
    return"not a fake news"
  
def manual_testing2(news):
  testing_news={"text":[news]}
  new_def_test=pd.DataFrame(testing_news)
  new_def_test["text"]=new_def_test["text"].apply(wordopt)
  new_x_test=new_def_test["text"]
  new_xv_test=vectorization.transform(new_x_test)
  import time
  with st.spinner('Wait for it...'):
    time.sleep(10)
  st.success('Done!')
  pred_DT=DT.predict(new_xv_test)
  score=(100*DT.score(xv_test,y_test))
  #return print("\n\nDT Prediction: {} ".format(output_label(pred_DT[0])))
  #s= st.text("output_label(pred_DT[0])")
  print("\n\nDT Prediction: {} ".format(output_label(pred_DT[0])))
  #print(type(s))
  print(str("DT Prediction: {} ".format(output_label(pred_DT[0]))))
  print("THE PREDICTION ACCURACY:",score)
  st.text(str("DT Prediction: {} ".format(output_label(pred_DT[0]))))  
  st.text("PREDICTION ACCURACY:")
  st.text(score)                                                               
# news = str(input())


# manual_testing(news)
def manual_testing3(news):
  testing_news={"text":[news]}
  new_def_test=pd.DataFrame(testing_news)
  new_def_test["text"]=new_def_test["text"].apply(wordopt)
  new_x_test=new_def_test["text"]
  new_xv_test=vectorization.transform(new_x_test)
  import time
  with st.spinner('Wait for it...'):
    time.sleep(10)
  st.success('Done!')
  pred_LR=LR.predict(new_xv_test)
  score1=(100*LR.score(xv_test,y_test))
  #s= st.text("output_label(pred_LR[0])")
  print("\n\nLR Prediction: {} ".format(output_label(pred_LR[0])))
  #print(type(s))
  print(str("LR Prediction: {} ".format(output_label(pred_LR[0]))))
  print("THE PREDICTION ACCURACY:",score1)
  # if s==:
    #  st.text("FAKE NEWS")
  # else:
    #  st.text("Real news")

  st.text(str("LR Prediction: {} ".format(output_label(pred_LR[0]))))
  st.text("PREDICTION ACCURACY:")
  st.text(score1)
    

def manual_testing4(news):
  testing_news={"text":[news]}
  new_def_test=pd.DataFrame(testing_news)
  new_def_test["text"]=new_def_test["text"].apply(wordopt)
  new_x_test=new_def_test["text"]
  new_xv_test=vectorization.transform(new_x_test)
  import time
  with st.spinner('Wait for it...'):
    time.sleep(10)
  st.success('Done!')
  pred_GB=GB.predict(new_xv_test)
  score2=(100*GB.score(xv_test,y_test))
  #return print("\n\nGB Prediction: {}".format(output_label(pred_GB[0])))
  #s= st.text("output_label(pred_GB[0])")
  print("\n\nGB Prediction: {} ".format(output_label(pred_GB[0])))
  #print(type(s))
  print(str("GB Prediction: {} ".format(output_label(pred_GB[0]))))
  print("THE PREDICTION ACCURACY:",score2)
  st.text(str("GB Prediction: {} ".format(output_label(pred_GB[0]))))
  st.text("PREDICTION ACCURACY:")
  st.text(score2)

def manual_testing1(news):
  testing_news={"text":[news]}
  new_def_test=pd.DataFrame(testing_news)
  new_def_test["text"]=new_def_test["text"].apply(wordopt)
  new_x_test=new_def_test["text"]
  new_xv_test=vectorization.transform(new_x_test)
  with st.spinner('Wait for it...'):
    time.sleep(10)
  st.success('Done!')
  pred_RF=RF.predict(new_xv_test)
  score3=(100*RF.score(xv_test,y_test))
  #s= st.text("output_label(pred_RF[0])")
  print("\n\nRF Prediction: {} ".format(output_label(pred_RF[0])))
  #print(type(s))
  print(str("RF Prediction: {} ".format(output_label(pred_RF[0]))))
  print("THE PREDICTION ACCURACY:",score3)
  st.text(str("RF Prediction: {} ".format(output_label(pred_RF[0]))))
  st.text("PREDICTION ACCURACY:")
  st.text(score3)

st.title ("FAKE AND REAL NEWS DETECTION")
st.markdown("<span style='color:black'>Our platform helps you differentiate between real and fake news.We use advanced algorithms and machine learning to quickly analyze news articles for authenticity. Our goal is to provide accurate and reliable news so that you can make informed decisions. You can easily check the authenticity of an article by entering the text, and we also offer a curated collection of articles on various topics. Our platform promotes responsible journalism and is accessible to all. Join us in the fight against fake news and stay informed with our platform.</span>",unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1586075010923-2dd4570fb338?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



#with st.form(key="form1"):
    #col1,col2,col3=st.columns(3)
val = st.text_area("Input News ")




st.text("")
st.text("")
st.text("Select the  Algorithm with which you want to perform ")
# def change():
#         print("selected")

# state=st.checkbox("Random Forest",value=False,on_change=change)
# state1=st.checkbox("Logical Regression",value=False,on_change=change)
# state2=st.checkbox("Decision Tree",value=False,on_change=change)
# state3=st.checkbox("Gradient Boosting",value=False,on_change=change)


col1, col2, col3,col4 = st.columns(4)

with col1:
    button1 = st.button('Random Forest')

with col2:
    button2 = st.button('Decision Tree')

with col3:
    button3 = st.button('Logical Regression')
with col4:
    button4 = st.button('Gradient Boosting classifier')

if button1:
    # Do something...
    print("you have choosen Random Forest")
    
    RF= RandomForestClassifier(random_state=0)
    RF.fit(xv_train, y_train)
    pred_rf=RF.predict(xv_test)
    RF.score(xv_test,y_test)
    print(classification_report(y_test,pred_rf))
    # manual_testing3(val)
    testing_news={"text":[val]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_rf=RF.predict(new_xv_test)
    s5 = str(manual_testing1(val))
    # print(s5)

if button2:
    # Do something...
    print("you have choosen decision tree")
    # from sklearn.tree import DecisionTreeClassifier
    DT = DecisionTreeClassifier(random_state=0)
    DT.fit(xv_train, y_train)
    pred_dt=DT.predict(xv_test)
    DT.score(xv_test,y_test)
    print(classification_report(y_test,pred_dt))
    testing_news={"text":[val]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_dt=DT.predict(new_xv_test)
    s6 = str(manual_testing2(val))
    # print(s6)

if button3:
    # Do something...
    print("choosen logical regression")
    from sklearn.linear_model import LogisticRegression 
    LR= LogisticRegression(random_state=0)
    LR.fit(xv_train, y_train)
    pred_lr=LR.predict(xv_test)
    LR.score(xv_test,y_test)
    print(classification_report(y_test,pred_lr))
    # manual_testing3(val)
    testing_news={"text":[val]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    s3 = str(manual_testing3(val))

if button4:
    #do somethiing
    print("choosen gradient boosting")
    from sklearn.ensemble import GradientBoostingClassifier
    GB=GradientBoostingClassifier(random_state=0)
    GB.fit(xv_train,y_train)
    predit_gb=GB.predict(xv_test)
    GB.score(xv_test,y_test)
    print(classification_report(y_test,predit_gb))
    # manual_testing3(val)
    testing_news={"text":[val]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    predit_gb=GB.predict(new_xv_test)
    s4 = str(manual_testing4(val))
    # print(s4)
    
    
# st.text("")
# st.text("")
# col1, col2, col3 , col4, col5 = st.columns(5)

# with col1:
#     pass
# with col2:
#     pass
# with col4:
#     pass
# with col5:
#     pass
# with col3 :
#     center_button = st.button('PREDICT')
# if center_button:
#    if button1:
#       a=manual_testing1(val)
#       print(a)
#    if button2:
#       b=manual_testing2(val)
#       print(b)
#    if button3:
#       c=manual_testing3(val)
#       print(c)
#    if button4:
#       d=manual_testing4(val)
#       print(d)
      
            
      
      

# st.button(label="PREDICT")
# if submit:
#       print("Entered News is:",val)
#       st.text(val)
#       if state:
#        print("You have choosen random forest")
#       if state1:
#         print("You have choosen Logical Regression")
#       if state2:
#         print("you have choosen Decision Tree")
#       if state3:
#         print("you have choosen Gradient Boosting")


# state5=st.text("RESULT:")
# if state5:
   


    