import utils
import pandas as pd
from math import log
import numpy as np
import argparse

training = pd.read_csv("data/training_set.tsv", header=0,  delimiter="\t", quoting=3)


def get_questions_tf_idf(training_data):
    """ indexing wiki pages:
    returns {document1:{word1:tf, word2:tf ...}, ....},
            {word1: idf, word2:idf, ...}"""
    question = training_data['question']

    ques_tf = {}
    idf = {}
    vocab = set()
    index = 0
    for ques in question:
        dd = {}
        w_count = 0
        lst = utils.tokenize(ques)
        for w in lst:
            vocab.add(w)
            dd.setdefault(w, 0)
            dd[w] +=1
            w_count += 1

        for k, v in dd.iteritems():
            dd[k] = 1. * v / w_count

        ques_tf[index] = dd
        index += 1

    for w in list(vocab):
        docs_with_w = 0
        for path, doc_tf in ques_tf.iteritems():
            if w in doc_tf:
                docs_with_w += 1
        idf[w] = log(len(ques_tf)/docs_with_w)

    return ques_tf, idf


def get_ques_importance_for_question(question, docs_tf, word_idf, max_docs = None):
    question_words = set(utils.tokenize(question))
    # go through each article

    doc_importance = []
    for doc, doc_tf in docs_tf.iteritems():

        doc_imp = 0
        for w in question_words:
            if w in doc_tf:
                doc_imp += doc_tf[w]  * word_idf[w]
        doc_importance.append((doc, doc_imp))

    # sort doc importance
    doc_importance = sorted(doc_importance, key=lambda x: x[1], reverse = True)
    if max_docs:
        return doc_importance[:max_docs]
    else:
        return doc_importance





def predict(data, docs_per_q):
    #index docs
    docs_tf, words_idf = get_questions_tf_idf(training)

    res = []

    for index, row in data.iterrows():
        #get answers words
        if index % 100 == 0:
            print 'processing', index, 'th questions'
        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))

        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0

        q = row['question']

        for d in zip(*get_ques_importance_for_question(q, docs_tf, words_idf, docs_per_q))[0]:

            an = utils.tokenize(training['correctAnswer'][d])
            ans = utils.tokenize(training['answerA'][d])
            if an.__contains__('b'):
                ans = utils.tokenize(training['answerB'][d])
            elif an.__contains__('c'):
                ans = utils.tokenize(training['answerC'][d])
            elif an.__contains__('d'):
                ans = utils.tokenize(training['answerD'][d])

            for w in w_A:
                if w in ans:
                    sc_A += 1
            for w in w_B:
                if w in ans:
                    sc_B += 1
            for w in w_C:
                if w in ans:
                    sc_C += 1
            for w in w_D:
                if w in ans:
                    sc_D += 1

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])


    return res


if __name__ == '__main__':
    # parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='test_set.tsv', help='file name with data')
    parser.add_argument('--docs_per_q', type=int, default= 100, help='number of docs to consider when ranking quesitons')
    parser.add_argument('--get_data', type=int, default= 0, help='flag to get wiki data for IR')
    args = parser.parse_args()

    # read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )
    # predict
    res = predict(data, 150)
    # save result
    pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("prediction.csv", index = False)


# datatrain = pd.read_csv('data/training_test_set.tsv', sep = '\t')
# re = datatrain["correctAnswer"]
# pre = predict(datatrain, 100)
# count = 0
# for i in xrange(0,len(re)):
#     if re[i] == pre[i]:
#         count += 1
#
# acc_train = float(count)/len(re)
# print acc_train
#

