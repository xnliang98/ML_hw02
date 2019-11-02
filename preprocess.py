import pandas as pd
import re

train_path = './dataset/sentiment/train.csv'
test_path = './dataset/sentiment/test.csv'
dev_path = './dataset/sentiment/dev.csv'


def clean_text(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    # punctuation
    text = re.sub(r"\+", "", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", "", text)
    text = re.sub(r"/", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"=", "", text)
    text = re.sub(r"\^", "", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"&", "", text)
    text = re.sub(r"\|", "", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", "", text)
    text = re.sub(r"\)", "", text)
    # symbol replacement
    # text = re.sub(r"&", " and ", text)
    # text = re.sub(r"\|", " or ", text)
    # text = re.sub(r"=", " equal ", text)
    # text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)
    # remove extra space
    text = ' '.join(text.split())
    reg = re.compile('<[^>]*>')
    text = reg.sub('', text)
    return text

def process(df, is_test=False):
    sentences = df.sentence.values
    labels = df.label.values
    # print(sentences[10].lower())
    # nlp = spacy.load('en')
    # doc = nlp(sentences[10])
    # print(doc.token)
    sens = []
    for s in sentences:
        s = clean_text(s)
        s_list = s.split(" ")
        if len(s_list) > 512:
            s_list = s_list[:512]
            s = " ".join(s_list)
        sens.append(s)
    df = pd.DataFrame({'sentence': sens, 'label': labels})
    return df



def main():
    train_df = pd.read_csv(train_path, header=0)
    train_df = process(train_df)
    train_df.to_csv(train_path, header=True, index=False)

    dev_df = pd.read_csv(dev_path, header=0)
    dev_df = process(dev_df)
    dev_df.to_csv(dev_path, header=True, index=False)

    test_df = pd.read_csv(test_path, header=0)
    test_df = process(test_df)
    test_df.to_csv(test_path, header=True, index=False)

if __name__ == "__main__":
    main()