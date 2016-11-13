import pandas as pd
from review_to_words import review_to_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# first we import the data
train = pd.read_csv("training_set.tsv", header=0,  delimiter="\t", quoting=3)

print train.shape

id = train["id"]
questions = train["question"]
correctAnswer = train["correctAnswer"]
answerA = train["answerA"]
answerB = train["answerB"]
answerC = train["answerC"]
answerD = train["answerD"]

num_questions = id.size

clean_questions = []
clean_answerA = []
clean_answerB = []
clean_answerC = []
clean_answerD = []
clean_A = []
clean_B = []
clean_C = []
clean_D = []
correctAnswer_word = []
wrongAnswer_word = []

for i in xrange(0, num_questions):
    clean_questions.append( review_to_words(questions[i]))
    clean_A.append( review_to_words(answerA[i]))
    clean_B.append( review_to_words(answerB[i]))
    clean_C.append( review_to_words(answerC[i]))
    clean_D.append( review_to_words(answerD[i]))
    AA = set(clean_A[i].split(" ")) - set(clean_B[i].split(" ")) - set(clean_C[i].split(" ")) - set(clean_D[i].split(" "))
    BB = set(clean_B[i].split(" ")) - set(clean_C[i].split(" ")) - set(clean_D[i].split(" ")) - set(clean_A[i].split(" "))
    CC = set(clean_C[i].split(" ")) - set(clean_B[i].split(" ")) - set(clean_A[i].split(" ")) - set(clean_D[i].split(" "))
    DD = set(clean_D[i].split(" ")) - set(clean_B[i].split(" ")) - set(clean_C[i].split(" ")) - set(clean_A[i].split(" "))
    clean_answerA.append(" ".join(AA))
    clean_answerB.append(" ".join(BB))
    clean_answerC.append(" ".join(CC))
    clean_answerD.append(" ".join(DD))


print clean_questions[0].split(" ")

for i in xrange(0, num_questions):
    if correctAnswer[i] == "A":
        correctAnswer_word.append(clean_answerA[i])
        wrongAnswer_word.append(clean_answerB[i] + " " + clean_answerC[i] + " " + clean_answerD[i])
    if correctAnswer[i] == "B":
        correctAnswer_word.append(clean_answerB[i])
        wrongAnswer_word.append(clean_answerA[i] + " " + clean_answerC[i] + " " + clean_answerD[i])
    if correctAnswer[i] == "C":
        correctAnswer_word.append(clean_answerC[i])
        wrongAnswer_word.append(clean_answerB[i] + " " + clean_answerA[i] + " " + clean_answerD[i])
    if correctAnswer[i] == "D":
        correctAnswer_word.append(clean_answerD[i])
        wrongAnswer_word.append(clean_answerB[i] + " " + clean_answerC[i] + " " + clean_answerA[i])


print correctAnswer_word.__len__()
print wrongAnswer_word.__len__()
print wrongAnswer_word


# now convert each question to trainable bag of word data
# questions_train = []
# for i in xrange(0, num_questions):
#     current_true = clean_questions[i] + " " + correctAnswer_word[i]
#     current_wrong = clean_questions[i] + " " + wrongAnswer_word[i]
#     questions_train.append(current_true)
#     questions_train.append(current_wrong)

# now convert each question to trainable bag of word data
questions_train = []
for ij in xrange(0, num_questions):
    q_set = clean_questions[ij].split(" ")
    ca_set = correctAnswer_word[ij].split(" ")
    wa_set = wrongAnswer_word[ij].split(" ")
    current_true_ques = ""
    current_wrong_ques = ""
    for j in xrange(0,len(q_set)):
        for k in xrange(0,len(ca_set)):
            current_word = q_set[j] + ca_set[k]
            current_true_ques += current_word + " "
        for l in xrange(0,len(wa_set)):
            current_word = q_set[j] + wa_set[l]
            current_wrong_ques += current_word + " "
    questions_train.append(current_true_ques)
    questions_train.append(current_wrong_ques)

print len(questions_train)

label = []
for ss in xrange(0, 5000):
    if ss % 2 == 1:
        label.append(0)
    else:
        label.append(1)

print label
print len(label)

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=60000)

train_data_features = vectorizer.fit_transform(questions_train)

train_data_features = train_data_features.toarray()

print "Training the random forest..."
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit(train_data_features, label)

# read the test data
test = pd.read_csv("test_set.tsv", header=0, delimiter="\t", quoting=3)
test_id = test["id"]
test_questions = test["question"]
test_answerA = test["answerA"]
test_answerB = test["answerB"]
test_answerC = test["answerC"]
test_answerD = test["answerD"]
#
clean_test_questions = []
clean_test_answerA = []
clean_test_answerB = []
clean_test_answerC = []
clean_test_answerD = []

for iter in xrange(0, len(test_id)):
    clean_test_questions.append( review_to_words(test_questions[iter]))
    clean_test_answerA.append( review_to_words(test_answerA[iter]))
    clean_test_answerB.append( review_to_words(test_answerB[iter]))
    clean_test_answerC.append( review_to_words(test_answerC[iter]))
    clean_test_answerD.append( review_to_words(test_answerD[iter]))

# now convert each question to trainable bag of word data
questions_A = []
questions_B = []
questions_C = []
questions_D = []
for it in xrange(0, len(test_id)):
    questions_set = clean_test_questions[it].split(" ")
    A_set = clean_test_answerA[it].split(" ")
    B_set = clean_test_answerB[it].split(" ")
    C_set = clean_test_answerC[it].split(" ")
    D_set = clean_test_answerD[it].split(" ")
    current_A = ""
    current_B = ""
    current_C = ""
    current_D = ""
    for jj in xrange(0,len(questions_set)):
        for v in xrange(0,len(A_set)):
            current_word = questions_set[jj] + A_set[v]
            current_A += current_word + " "
        for q in xrange(0,len(B_set)):
            current_word = questions_set[jj] + B_set[q]
            current_B += current_word + " "
        for m in xrange(0,len(C_set)):
            current_word = questions_set[jj] + C_set[m]
            current_C += current_word + " "
        for n in xrange(0,len(D_set)):
            current_word = questions_set[jj] + D_set[n]
            current_D += current_word + " "
    questions_A.append(current_A)
    questions_B.append(current_B)
    questions_C.append(current_C)
    questions_D.append(current_D)

# =======================bag of word for one word===========
# now convert each question to trainable bag of word data
# questions_A = []
# questions_B = []
# questions_C = []
# questions_D = []
# i = 0
# for i in xrange(0, len(test_id)):
#     A_set = clean_test_questions[i] + " " + clean_test_answerA[i]
#     B_set = clean_test_questions[i] + " " + clean_test_answerB[i]
#     C_set = clean_test_questions[i] + " " + clean_test_answerC[i]
#     D_set = clean_test_questions[i] + " " + clean_test_answerD[i]
#
#     questions_A.append(A_set)
#     questions_B.append(B_set)
#     questions_C.append(C_set)
#     questions_D.append(D_set)

# Get a bag of words for the test set, and convert to a numpy array
test_data_featuresA = vectorizer.transform(questions_A)
test_data_featuresB = vectorizer.transform(questions_B)
test_data_featuresC = vectorizer.transform(questions_C)
test_data_featuresD = vectorizer.transform(questions_D)
test_data_featuresA = test_data_featuresA.toarray()
test_data_featuresB = test_data_featuresB.toarray()
test_data_featuresC = test_data_featuresC.toarray()
test_data_featuresD = test_data_featuresD.toarray()

# Use the random forest to make sentiment label predictions
resultA = forest.predict_proba(test_data_featuresA)
resultB = forest.predict_proba(test_data_featuresB)
resultC = forest.predict_proba(test_data_featuresC)
resultD = forest.predict_proba(test_data_featuresD)
A = [x[0] for x in resultA]
B = [x[0] for x in resultB]
C = [x[0] for x in resultC]
D = [x[0] for x in resultD]


result = []
for ite in xrange(0,len(test_id)):
    current_ans = "A"
    if 1-A[ite] < 1-B[ite]:
        current_ans = "B"
    if 1-B[ite] < 1-C[ite]:
        current_ans = "C"
    if 1-C[ite] < 1-D[ite]:
        current_ans = "D"
    result.append(current_ans)


# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame(data={"id":test["id"], "correctAnswer":result})

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)