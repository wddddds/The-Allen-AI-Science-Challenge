from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.fields import *
import os
import utils
from whoosh.qparser import QueryParser
import pandas as pd


data_dir = 'data/wiki_data'
# schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
# if not os.path.exists("index"):
#     os.mkdir("index")
# ix = create_in("index", schema)

ix = open_dir("index")
writer = ix.writer()
# write the articles into index
for fname in os.listdir(data_dir):
    path = os.path.join(data_dir, fname)
    cont = []
    for index, line in enumerate(open(path)):
        cont += utils.tokenize(line)
    content = " ".join(cont)
    writer.add_document(title=unicode(fname), content=unicode(content))
writer.commit()


def rank_question_by_whoosh(question, lim):
    res = []
    question_words = utils.tokenize(question)
    ques_str = unicode(" ".join(question_words))
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        query = parser.parse(ques_str)
        results = searcher.search(query, limit = lim)
        results.fragmenter.charlimit = 100000
        res.append(results)
        print results[0]

    return res


data = pd.read_csv('data/training_test_set.tsv', sep = '\t')

for ind, row in data.iterrows():
    res = rank_question_by_whoosh(row['question'], 5)
    print res


