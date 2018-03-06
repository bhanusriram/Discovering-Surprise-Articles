from os import listdir
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from collections import defaultdict
import csv


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


if __name__ == '__main__':
    doc_complete = []
    input_path = "C:/Users/doula/Desktop/KDD Final Project Workspace/diabetes/"
    output_path = "C:/Users/doula/Desktop/KDD Final Project Workspace/"
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    doc_names_list = []
    for current_file in listdir(input_path):
        doc_text = ''
        input_file = open(input_path + current_file, 'r', encoding="utf8")
        current_doc = input_file.readlines()
        doc_text = str(current_doc)
        doc_complete.append(doc_text)
        input_file.close()

    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    lda_model = Lda(doc_term_matrix, num_topics=30, id2word=dictionary, passes=50)
    topic_list = lda_model.print_topics(num_topics=30, num_words=10)
    doc_topic_distribution = [lda_model[dictionary.doc2bow(doc)] for doc in doc_clean]

    doc_topic_distribution = defaultdict(dict)
    for current_file in listdir(input_path):
        for index in range(30):
            doc_topic_distribution[current_file][index] = 0

    for current_file in listdir(input_path):
        doc_text = ''
        input_file = open(input_path + current_file, 'r', encoding="utf8")
        current_doc = input_file.readlines()
        doc_bow = dictionary.doc2bow(clean(str(current_doc)).split())
        doc_topic_distr = lda_model[doc_bow]
        for d in doc_topic_distr:
            doc_topic_distribution[current_file][d[0]] = round(d[1], 1)

    topic_list.insert(0, '')
    with open(output_path + 'topics_distribution_rounded.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(topic_list)
        for k, v in doc_topic_distribution.items():
            writer.writerow([k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13],
                            v[14], v[15], v[17], v[16], v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26],
                            v[27], v[28], v[29]])
