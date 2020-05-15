from random import choice
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore') ##### ignores the warning 

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

"""
Required libraries can be found in the "requirement.txt".
type "pip install -r requirement.txt" in cmd to install require repositories at once.

"""

# print(stopwords.words("english"))

################### Reading corpus

with open ('chatbot-student.txt','r',encoding='utf8',errors='ignore') as f:
    raw = f.read().lower()

################### Tokenisation

sent_tokens = nltk.sent_tokenize(raw) ########## to lists of sentences
word_tokens = nltk.word_tokenize(raw) ########## to lists of words

# print("sent_tokens: ", sent_tokens)
# print("word_tokens: ", word_tokens)

############### Preprocessing

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punt_dict = {
    ord(punct):None for punct in string.punctuation
    }

# print(remove_punt_dict)

def LemNormalize(text):
    # print('text before: ', text)
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    text = ' '.join(text)
    # print("text after: ", text)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punt_dict) )) ############ Removes all the punctuation marks from text

##################### Keyword matching
GREETING_INPUTS = ("hello","hi","greetings","sup","what's up","hey")
GREETING_RESPONSES = ("hi","hey :)","*nods*","hi there","hello, thank you for asking","i'm glad you are talking to me")


def greetings(sentence):
    for word in sentence.split():
        word = word.lower().translate(remove_punt_dict)
        
        # print("Greeting recieved: ", word)
        
        if word in GREETING_INPUTS:
            return choice(GREETING_RESPONSES)



#################### Greeting Response

def response(user_response):
    robo_response = ''

    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # print("tfidf: ", tfidf)
    
    vals = cosine_similarity(tfidf[-1],tfidf)
    
    # print("vals: ", vals)
    
    idx = vals.argsort()[0][-2]

    # print("idx: ", idx)

    flat = vals.flatten()
    flat.sort()

    # print("flat: ", flat)

    req_tfidf = flat[-2]

    # print("req_tfidf: ", req_tfidf)

    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response



################# Initiating Robot

flag = True
print("BOT: I'm ROBOT. I shall answer all quries possible in my knowledge. if you want to exit, type BYE")

while (flag == True):
    # print("YOU: ", end='')
    user_response = input('YOU: ')
    user_response = user_response.lower()

    if (user_response != 'bye'):
        if ('thanks' in user_response or 'thank you' in user_response):
            flag=False
            print('BOT: You are welcome..')
        else:
            if (greetings(user_response) != None):
                print("BOT: "+greetings(user_response))
            else:
                print("BOT: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("BOT: Bye! Take care..")           
            
