import argparse
import utils
import numpy as np
import pandas as pd
from review_to_words import review_to_words

# urls  to get topics
ck12_url_topic = ['https://www.ck12.org/earth-science/', 'http://www.ck12.org/life-science/',
                  'http://www.ck12.org/physical-science/', 'http://www.ck12.org/biology/',
                  'http://www.ck12.org/chemistry/', 'http://www.ck12.org/physics/']
wiki_docs_dir = 'data/wiki_data - training word + textbook'



def get_wiki_docs():
    # get keywords
    ck12_keywords = set()
    for url_topic in ck12_url_topic:
        keywords = utils.get_keyword_from_url_topic(url_topic)
        for kw in keywords:
            ck12_keywords.add(kw)

    # get and save wiki docs
    utils.get_save_wiki_docs(ck12_keywords, wiki_docs_dir)


def get_wiki_docs2():
    # download wikipedia document about appeared words in test questions
    train = pd.read_csv("training_set.tsv", header=0,  delimiter="\t", quoting=3)
    questions = train["question"]
    clean_questions = ""
    num_questions = train["id"].size
    for i in xrange(0, num_questions):
        clean_questions = clean_questions + " " + review_to_words(questions[i])

    x = [u'']
    x = set(x)
    keywords_train = set(clean_questions.split(" ")) - x
    utils.get_save_wiki_docs(keywords_train, wiki_docs_dir)


def get_wiki_docs3():
    # download wikipedia document about appeared words in test questions
    train = pd.read_csv("test_set.tsv", header=0,  delimiter="\t", quoting=3)
    questions = train["question"]
    clean_questions = ""
    num_questions = train["id"].size
    for i in xrange(0, num_questions):
        clean_questions = clean_questions + " " + review_to_words(questions[i])

    x = [u'']
    x = set(x)
    keywords_train = set(clean_questions.split(" ")) - x
    utils.get_save_wiki_docs(keywords_train, wiki_docs_dir)


def get_wiki_docs4():
    # download wikipedia document about appeared words in test questions
    train = pd.read_csv("science-wikipedia-articles.txt",   delimiter="\t", quoting=3)
    questions = train["environmental change"]
    clean_questions = []
    for i in xrange(0, len(questions)-1):
         clean_questions.append( review_to_words(questions[i]))

    x = [u'']
    x = set(x)

    keywords_train = set(clean_questions) - x

    utils.get_save_wiki_docs(keywords_train, wiki_docs_dir)


def predict(data, docs_per_q):
    #index docs
    docs_tf, words_idf = utils.get_docstf_idf(wiki_docs_dir)

    res = []

    for index, row in data.iterrows():
        #get answers words

        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))

        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0

        q = row['question']

        if index % 100 == 0:
            print 'processing', index, 'th questions'

        for d in zip(*utils.get_docs_importance_for_question(q, docs_tf, words_idf, docs_per_q))[0]:


            for w in w_A:
                if w in docs_tf[d]:
                    sc_A += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_B:
                if w in docs_tf[d]:
                    sc_B += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_C:
                if w in docs_tf[d]:
                    sc_C += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_D:
                if w in docs_tf[d]:
                    sc_D += 1. * docs_tf[d][w] * words_idf[w]

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])


    return res


def predict_bm25(data, docs_per_q):
    #index docs
    docs_tf, words_idf, ave_length,doc_length = utils.get_docstf_idf(wiki_docs_dir)

    res = []

    for index, row in data.iterrows():
        #get answers words

        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))

        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0

        q = row['question']

        if index % 100 == 0:
            print 'processing', index, 'th questions'

        for d in zip(*utils.get_docs_importance_for_question_bm25(q, docs_tf, words_idf, ave_length, doc_length, docs_per_q))[0]:


            for w in w_A:
                if w in docs_tf[d]:
                    sc_A += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_B:
                if w in docs_tf[d]:
                    sc_B += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_C:
                if w in docs_tf[d]:
                    sc_C += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_D:
                if w in docs_tf[d]:
                    sc_D += 1. * docs_tf[d][w] * words_idf[w]

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])


    return res


# def predict(data, docs_per_q):
#     index docs
    # docs_tf, words_idf = utils.get_docstf_idf(wiki_docs_dir)
    #
    # res = []
    #
    # vocab = set()
    #
    # for index, row in data.iterrows():
    #     get answers words
        # w_a = utils.tokenize(row['answerA'])
        # w_b = utils.tokenize(row['answerB'])
        # w_c = utils.tokenize(row['answerC'])
        # w_d = utils.tokenize(row['answerD'])
        # wa = set(w_a)
        # wb = set(w_b)
        # wc = set(w_c)
        # wd = set(w_d)
        #
        # tf_A = utils.get_answer_tf(w_a)
        # tf_B = utils.get_answer_tf(w_b)
        # tf_C = utils.get_answer_tf(w_c)
        # tf_D = utils.get_answer_tf(w_d)
        #
        # w_A = wa-wb-wc-wd
        # w_B = wb-wa-wc-wd
        # w_C = wc-wb-wa-wd
        # w_D = wd-wb-wc-wa
        #
        # dot_A = 0
        # dot_B = 0
        # dot_C = 0
        # dot_D = 0
        #
        # abs_A =0
        # abs_B =0
        # abs_C =0
        # abs_D =0
        #
        # abs_doc_A = 0
        # abs_doc_B = 0
        # abs_doc_C = 0
        # abs_doc_D = 0
        #
        # sc_A = 0
        # sc_B = 0
        # sc_C = 0
        # sc_D = 0
        #
        # q = row['question']
        #
        # for d in zip(*utils.get_docs_importance_for_question(q, docs_tf, words_idf, docs_per_q))[0]:
        #     for w in w_A:
        #         if w in docs_tf[d]:
        #             tfidf_doc = 1. * docs_tf[d][w] * words_idf[w]
        #             tfidf_A = 1. * tf_A[w] * words_idf[w]
        #             ele_dot = 1. * tfidf_A * tfidf_doc
        #             dot_A += ele_dot
        #             abs_A += tfidf_A ** 2
        #             abs_doc_A += tfidf_doc ** 2
        #             sc_A =+ dot_A / (float(np.sqrt(abs_A) * np.sqrt(abs_doc_A)) + 0.001)
        #     for w in w_B:
        #         if w in docs_tf[d]:
        #             tfidf_doc = 1. * docs_tf[d][w] * words_idf[w]
        #             tfidf_B = 1. * tf_B[w] * words_idf[w]
        #             ele_dot = 1. * tfidf_B * tfidf_doc
        #             dot_B += ele_dot
        #             abs_B += tfidf_B ** 2
        #             abs_doc_B += tfidf_doc ** 2
        #             sc_B += dot_B / (float(np.sqrt(abs_B) * np.sqrt(abs_doc_B)) + 0.001)
        #     for w in w_C:
        #         if w in docs_tf[d]:
        #             tfidf_doc = 1. * docs_tf[d][w] * words_idf[w]
        #             tfidf_C = 1. * tf_C[w] * words_idf[w]
        #             ele_dot = 1. * tfidf_C * tfidf_doc
        #             dot_C += ele_dot
        #             abs_C += tfidf_C ** 2
        #             abs_doc_C += tfidf_doc ** 2
        #             sc_C += dot_C / (float(np.sqrt(abs_C) * np.sqrt(abs_doc_C)) + 0.001)
        #     for w in w_D:
        #         if w in docs_tf[d]:
        #             tfidf_doc = 1. * docs_tf[d][w] * words_idf[w]
        #             tfidf_D = 1. * tf_D[w] * words_idf[w]
        #             ele_dot = 1. * tfidf_D * tfidf_doc
        #             dot_D += ele_dot
        #             abs_D += tfidf_D ** 2
        #             abs_doc_D += tfidf_doc ** 27
        #             sc_D += dot_D / (float(np.sqrt(abs_D) * np.sqrt(abs_doc_D)) + 0.001)
        #
        # res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])
    #
    # return res

if __name__ == '__main__':
    # parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='test_set.tsv', help='file name with data')
    parser.add_argument('--docs_per_q', type=int, default= 50, help='number of docs to consider when ranking quesitons')
    parser.add_argument('--get_data', type=int, default= 0, help='flag to get wiki data for IR')
    args = parser.parse_args()

    if args.get_data:
        get_wiki_docs()

    # read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )
    # predict
    res = predict(data, args.docs_per_q)
    # save result
    pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("prediction.csv", index = False)

    # compute the accuracy on training set
    # training = pd.read_csv("training_set.tsv", header=0,  delimiter="\t", quoting=3)
    # re = training["correctAnswer"]
    # datatrain = pd.read_csv('data/training_set.tsv', sep = '\t')
    # pre = predict(datatrain, args.docs_per_q)
    #
    # count = 0
    # for i in xrange(0,len(re)):
    #     if re[i] == pre[i]:
    #         count += 1
    #
    # acc_train = float(count)/len(re)
    # print acc_train


