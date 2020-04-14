from dateutil import parser
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

def parse_date(x):
    
    try: 
        return parser.parse(x)
    
    except:
        return np.nan
    
def process_text(text, length=False, stem=False):
    
    try:
    
        stop_words = set(stopwords.words('english'))
    
    except:
        
        nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))
    if stem:    
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word.lower()) for word in text.split() if (word.isalpha()) and (word not in stop_words)]
    else:
        tokens = [word.lower() for word in text.split() if (word.isalpha()) and (word not in stop_words)]

    cleaned_text = ' '.join(tokens)

    if length:
        length_of_text = len(tokens)
        return cleaned_text,length_of_text
    else:
        return cleaned_text
    
def csv_to_s3(df, prefix,filename, bucket, header=False):
    
    df.to_csv(filename, header, index=False)
    sagemaker_session.upload_data(bucket=bucket, key_prefix=prefix, path=filename)