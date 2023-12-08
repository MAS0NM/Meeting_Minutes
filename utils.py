import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(context):
    '''
        format of context: [sentence, sentence]
        ->
        format of output: [[word, word, word], [word, word], ...]
    '''
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    context = [[lemmatizer.lemmatize(tok) for tok in nltk.word_tokenize(sentence) if tok not in stop_words and tok not in punct] for sentence in context]
    return context
    

def cosine_similarity(vec1, vec2):
    
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def IoU(ls1, ls2):
    set1, set2 = set(ls1), set(ls2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)