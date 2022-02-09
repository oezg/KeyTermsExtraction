from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from lxml import etree
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
punctuation = set(punctuation)
stopwords = set(stopwords.words('english'))
trash = stopwords | punctuation

filename = "news.xml"
tree = etree.parse(filename)
root = tree.getroot()
corpus = root[0]
dataset = []
headlines = []

for news in corpus:
    headlines.append(news[0].text)
    tokens = word_tokenize(news[1].text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    lemmas = [lemma for lemma in lemmas if lemma not in trash]
    lemmas = [lemma for lemma in lemmas if pos_tag([lemma])[0][1] == "NN"]
    dataset.append(" ".join(lemmas))

tfidf_matrix = vectorizer.fit_transform(dataset).toarray()
terms = vectorizer.get_feature_names_out()
common_words = []

for tfidf_scores in tfidf_matrix:
    word_scores = {term: score for term, score in zip(terms, tfidf_scores)}
    words = sorted(word_scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    common_words.append(" ".join([word for word, score in words[:5]]))

for headline, result in zip(headlines, common_words):
    print(headline, result, sep=':\n', end='\n'*2)

