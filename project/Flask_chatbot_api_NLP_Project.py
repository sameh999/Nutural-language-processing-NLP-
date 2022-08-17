
# coding: utf-8

# In[16]:

get_ipython().system('pip install openpyxl')


# In[26]:

import sklearn
print(sklearn.__version__)
get_ipython().system('pip uninstall scikit-learn')
get_ipython().system('pip install scikit-learn')


# In[27]:

import re
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import json
import pandas as pd


# In[29]:

def remove_punct(book):
    # Remove Punctuation using regex
    return re.sub(r'[^\w\s]','', book)

def remove_numbers(book):
    # Remove Punctuation using regex
    return re.sub('[^A-Za-z]+', ' ', book)

def remove_stopwords(book):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    return pattern.sub('', book.lower())

def stemming(tokens):
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]

def tokenizer(text):
    return word_tokenize(text)

def case_fold(tokens):
    return [word.lower() for word in tokens]


# In[25]:
#
# df = pd.read_excel('drug_data.xlsx', names=['Category', 'Medicine_name', 'Ther_area',
#        'International non-proprietary name (INN) / common name',
#        'Active_substance', 'Product number', 'Patient safety',
#        'Authorisation status', 'ATC code', 'Additional monitoring', 'Generic',
#        'Biosimilar', 'Conditional approval', 'Exceptional circumstances',
#        'Accelerated assessment', 'Orphan medicine',
#        'Marketing authorisation date',
#        'Date of refusal of marketing authorisation',
#        'company_name',
#        'pharm_group', 'Vet pharmacotherapeutic group',
#        'Date of opinion', 'Decision date', 'Revision number',
#        'description', 'Species', 'ATCvet code', 'First published',
#        'Revision date', 'URL'])

# df.drop(columns=['Category',  'International non-proprietary name (INN) / common name',
#        'Active_substance', 'Product number', 'Patient safety',
#        'Authorisation status', 'ATC code', 'Additional monitoring', 'Generic',
#        'Biosimilar', 'Conditional approval', 'Exceptional circumstances',
#        'Accelerated assessment', 'Orphan medicine',
#        'Marketing authorisation date',
#        'Date of refusal of marketing authorisation',
#        'company_name',
#        'pharm_group', 'Vet pharmacotherapeutic group',
#        'Date of opinion', 'Decision date', 'Revision number',
#        'description', 'Species', 'ATCvet code', 'First published',
#        'Revision date'], inplace=True)

cat_df = pd.DataFrame(df.Ther_area.str.split(',').str[0].str.split(';').str[0].str.split(' ').str[0].value_counts())
df.Ther_area = df.Ther_area.str.split(',').str[0].str.split(';').str[0].str.split(' ').str[0]
df = df[df.Ther_area.isin(cat_df.Ther_area[:20].index)]
df.head()


# In[30]:

df = pd.read_excel('drug_data11.xlsx')


# In[31]:

result = df[df.Ther_area == 'Diabetes'][['Medicine_name','URL']][:3]


# In[36]:

from flask import Flask, jsonify, request, render_template
from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[35]:


app = Flask("name")

import pickle
nltk.download('stopwords')
nltk.download('punkt')

filename = 'finalized_model.sav'
# load the model from disk
models = pickle.load(open(filename, 'rb'))

model = models['model']
lenc = models['label_encoder']
tf_idf = models['tf_idf']
tfidf_obj_names = models['tf_idf_names']

@app.route('/', methods = ['GET', 'POST', 'DELETE'])
def home1(name='diabetes'):
    req = request.get_json(force=True)
    print(req['queryResult']['queryText'])
    name = req['queryResult']['queryText']
    punc = remove_punct(name)
    num = remove_numbers(punc)
    stop = remove_stopwords(num)
    tokens = tokenizer(stop)
    folded = case_fold(tokens)
    stem = stemming(folded)
    desc = ' '.join(stem)
    tf_idf1 = tf_idf.transform([desc])
    pred = model.predict(pd.DataFrame(tf_idf1.toarray(), columns=tfidf_obj_names))
    class_name = lenc.inverse_transform(pred)
    result = df[df.Ther_area == class_name[0]][['Medicine_name','URL']][:3]
    return {
        "fulfillmentText": "This is the suitable medicine for you " +result['Medicine_name'].values[0]+ "and you can check the link "+result['URL'].values[0] + " and i recommend this also for you \n"+
        result['Medicine_name'].values[1]+" : " +result['URL'].values[1]+"\n"+
        result['Medicine_name'].values[2]+" : " +result['URL'].values[2]

    }
    return json.loads('{"fulfillmentText":"'+ "I recommend these Drugs for U "+class_name[0]+'"}')
#     return class_names[loaded_model.predict(verctor_count.transform([name]))[0]]


# In[ ]:

app.run()


# In[ ]:




# In[ ]:
