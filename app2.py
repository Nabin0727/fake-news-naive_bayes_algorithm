#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


# In[2]:


app = Flask(__name__)
model = pickle.load(open('final_model.pkl','rb'))
vector = pickle.load(open('final_vector.pkl','rb'))


# In[3]:


translator = str.maketrans('', '', string.punctuation)
# remove_stop_words(text):
 #   stop_words = set(stopwords.words("english"))
  #  words = word_tokenize(text)
   # filtered_text = [word for word in words if word.lower() not in stop_words]
    #return " ".join(filtered_text)


# In[4]:


def fake_news_det(news):
    input_data = news 
#Preprcoessing
    input_data = input_data.lower()
    input_data = input_data.translate(translator)
    stop_words = stopwords.words('english')
    input_data = ' '.join([word for word in input_data.split() if word not in (stop_words)])
    #input_data = remove_stop_words(input_data)
    
    vectorized_input_data = vector.transform([input_data])
    prediction = model.predict(vectorized_input_data)
    return prediction


# In[5]:


#Flask routing 
@app.route("/")

def display():
    return render_template("Home.html")


# In[6]:


@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        news = request.form['message']
        pred = fake_news_det(news)
        print(pred)
        return render_template('Home.html', prediction = pred)
    else:
        return render_template('Home.html')


# In[ ]:





# In[7]:


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9000)


# In[ ]:




