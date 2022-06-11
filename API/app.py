from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
import emoji
import nltk
import os


app = Flask(__name__)
api = Api(app)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemma(tweet):
    
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer(preserve_case=False, 
                           strip_handles=True,
                           reduce_len=True)

    word_pos_tags = nltk.pos_tag(tokenizer.tokenize(tweet))
    t=[lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(t)

def removestopwords(tweet):
    t = [i for i in tweet.split() if i not in stopwords.words('english')]
    return ' '.join(t)

#cleaning the tweet
def clean_tweet(tweet):
    
    cleaned_tweet = tweet.replace("^RT[\s]+","").replace("https?:\/\/.*[\r\n]*","https")\
                        .replace("@[A-Za-z0-9]+","")\
                        .replace('[0-9]',' ').replace('[!@#$,?:.]', '').lower()
    #replace(emoji.get_emoji_regexp(),"")
    tokenized_tweet = lemma(cleaned_tweet)

    return removestopwords(tokenized_tweet)




folder = os.getcwd()
clf_path = '../models/TweetClassifier.pkl'
model = pickle.load(open(clf_path, 'rb'))

vec_path = '../models/TFIDFVectorizer.pkl'
vectorizer = pickle.load(open(vec_path, 'rb'))

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

categories = {0 : 'not_cyberbullying', 1 : 'gender', 2 : 'religion', 3 : 'age', 4 : 'ethnicity', 5 : 'other_cyberbullying'}

class classifyTweet(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        
        # Process the tweet
        cleaned_tweet = clean_tweet(str(user_query))
      
        
        # vectorize the user's query and make a prediction
        tweet_vectorized = vectorizer.transform(np.array([cleaned_tweet]))
        prediction = model.predict(tweet_vectorized)
        pred_proba = model.predict_proba(tweet_vectorized)

        # Output the category along with the score
        cat = categories[prediction[0]]

        # round the predict proba value and set to new variable
        confidence = round(max(pred_proba[0]), 3)

        # create JSON object
        output = {'prediction': cat, 'confidence': str(confidence)}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(classifyTweet, '/')


if __name__ == '__main__':
    app.run(debug=True)