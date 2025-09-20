import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import kagglehub
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

kagglehub.login()

path = kagglehub.dataset_download('grafstor/simple-dialogs-for-chatbot', 'dialogs.txt')
raw_data = nltk.data.load(path)

def preprocess(data):
    tweet = TweetTokenizer()
    tokens = tweet.tokenize(data)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens
              if word not in stop_words
              and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in set(tokens)]
    return tokens

questions_arr=[]
answers_arr=[]
data=[q.split('\t') for q in raw_data.split('\n')]
for sentence in data:
    questions_arr.append(sentence[0])
    answers_arr.append(sentence[1])

processed_data_questions = [preprocess(question) for question in questions_arr]
processed_data_answers = [preprocess(answer) for answer in answers_arr]