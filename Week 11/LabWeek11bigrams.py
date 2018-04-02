import nltk

# movie review sentences
from nltk.corpus import sentence_polarity
import random

## repeat the setup of the movie review sentences for classification
# for each sentence(document), get its words and category (positive/negative)
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

# get all words from all movie_reviews and put into a frequency distribution
#   note lowercase, but no stemming or stopwords
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
print(len(all_words))

# get the 1500 most frequently appearing keywords in the corpus
word_items = all_words.most_common(1500)
word_features = [word for (word,count) in word_items]

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

# training using naive Baysian classifier, training set is 90% of data
train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# evaluate the accuracy of the classifier
nltk.classify.accuracy(classifier, test_set)

# the accuracy result may vary since we randomized the documents

####   adding Bigram features   ####
# set up for using bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()

# create the bigram finder on all the words in sequence
all_words_list[:50]
finder = BigramCollocationFinder.from_words(all_words_list)

# define the top 500 bigrams using the chi squared measure
bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
print(bigram_features[:50])

# examples to demonstrate the bigram feature function definition
sent = ['Arthur','carefully','rode','the','brown','horse','around','the','castle']
sentbigrams = list(nltk.bigrams(sent))
print(sentbigrams)

# for a single bigram, test if it's in the sentence bigrams and format the feature name
bigram = ('brown','horse')
print(bigram in sentbigrams)
print('bigram({} {})'.format(bigram[0], bigram[1]))

# define features that include words as before 
# add the most frequent significant bigrams
# this function takes the list of words in a document as an argument and returns a feature dictionary
# it depends on the variables word_features and bigram_features
def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

# use this function to create feature sets for all sentences
bigram_featuresets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in documents]

# number of features for document 0
print(len(bigram_featuresets[0][0].keys()))

# features in document 0
print(bigram_featuresets[0][0])

# train a classifier and report accuracy
train_set, test_set = bigram_featuresets[1000:], bigram_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


###  POS tag counts
# using the default pos tagger in NLTK (the Stanford tagger)
print(sent)
print(nltk.pos_tag(sent))

# this function takes a document list of words and returns a feature dictionary
# it runs the default pos tagger (the Stanford tagger) on the document
#   and counts 4 types of pos tags to use as features
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features

# define feature sets using this function
POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in documents]
# number of features for document 0
print(len(POS_featuresets[0][0].keys()))

# the first sentenceprint(documents[0])# the pos tag features for this sentenceprint('num nouns', POS_featuresets[0][0]['nouns'])print('num verbs', POS_featuresets[0][0]['verbs'])print('num adjectives', POS_featuresets[0][0]['adjectives'])print('num adverbs', POS_featuresets[0][0]['adverbs'])

# train and test the classifier
train_set, test_set = POS_featuresets[1000:], POS_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


## other classifiers ##

# DecisionTree Classifier
classifier = nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff =0, support_cutoff =0)
nltk.classify.accuracy(classifier, test_set)

# MaxEnt Classifier
from nltk.classify import MaxentClassifier
classifier = nltk.MaxentClassifier.train(train_set, max_iter = 1)
nltk.classify.accuracy(classifier, test_set)

## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

# perform the cross-validation on the featuresets with word features and generate accuracy
num_folds = 5
cross_validation_accuracy(num_folds, featuresets)

