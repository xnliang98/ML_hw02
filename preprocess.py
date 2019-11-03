import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


train_path = './dataset/raw_data/labeledTrainData.tsv'
test_path = './dataset/raw_data/testData.tsv'


def clean_text(raw_review):
    REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer()
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)
    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    tokens = word_tokenize(lowercase_letters)
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return " ".join(meaningful_words)


def process(df, is_test=False):
    sentences = df.sentence.values
    labels = df.label.values
    sens = []
    for s in sentences:
        s = clean_text(s)
        sens.append(s)
    df = pd.DataFrame({'sentence': sens, 'label': labels})
    return df

def main():
    train_df = pd.read_csv(train_path, header=0, delimiter='\t')
    train_df = pd.DataFrame({'sentence': train_df.review.values, 'label': train_df.sentiment.values})
    train_df = process(train_df)
    train_df.to_csv('dataset/sentiment/train.csv', header=True, index=False)

    test_df = pd.read_csv(test_path, header=0, delimiter='\t')
    test_df = pd.DataFrame({'sentence': test_df.review.values})
    test_df['label'] = 0
    test_df = process(test_df)
    test_df.to_csv('dataset/sentiment/test.csv', header=True, index=False)

if __name__ == "__main__":
    main()