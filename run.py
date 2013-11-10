#!/usr/bin/python
# vim: set file-encoding=utf-8:

import sys
import math
import itertools
import operator
import cPickle

import maxent

from collections import defaultdict
from maxent import MaxentModel
from optparse import OptionParser

# |iterable| should yield lines.
def read_sentences(iterable):
    sentence = []
    for line in iterable:
        columns = line.rstrip().split()
        if len(columns) == 0 and len(sentence) > 0:
            yield sentence
            sentence = []
        if len(columns) > 0:
            sentence.append(columns)
    if len(sentence) > 0:
        yield sentence

# Computes (local) features for word at position |i| given that label for word
# at position |i - 1| is |previous_label|. You can pass any additional data
# via |data| argument.
MIN_WORD_FREQUENCY = 5
MIN_LABEL_FREQUENCY = 5

person_names = set()
def read_names(filename):
    inputFile = open(filename, 'r')
    global person_names
    for name in inputFile.readlines():
        name = name.upper()
        person_names.add(name.rstrip())

person_surnames = set()
def read_surnames(filename):
    inputFile = open(filename, 'r')
    global person_surnames
    for name in inputFile.readlines():
        name = name.upper()
        person_surnames.add(name.rstrip())

def upperLettersNumber(word):
    number = 0
    for letter in word:
        if str.istitle(letter):
            number = number + 1;
    return number

def compute_ne_features(data, words, poses, i, previous_label):
    # Condition on previous label.

    if previous_label != "O":
        yield "label-previous={0}".format(previous_label) 

    if data["word_frequencies"].get(words[i], 0) >= MIN_WORD_FREQUENCY:
        yield "word-current={0}".format(words[i])

    yield "pos={0}".format(poses[i])

    if (len(words[i]) > 3):
        yield "word[-2:]={0}".format(words[i][-2:])

    if (len(words[i]) > 4):
        yield "word[-3:]={0}".format(words[i][-3:])

    if (len(words[i]) > 5):
        yield "word[-4:]={0}".format(words[i][-4:])

    if (len(words[i]) > 3):
        yield "word[:2]={0}".format(words[i][:2])

    if (len(words[i]) > 4):
        yield "word[:3]={0}".format(words[i][:3])

    if (len(words[i]) > 3):
        yield "word[:-2]={0}".format(words[i][:-2])

    if (len(words[i]) > 4):
        yield "word[:-3]={0}".format(words[i][:-3])

    upperNumber = upperLettersNumber(words[i])
    if (upperNumber == len(words[i])):
        yield "is-ne=true"
    else:
        yield "is-abbreviation=false"

    # if (upperNumber * 2 >= len(words[i])):
    #     yield "is-abbreviation=true"
    # else:
    #     yield "is-abbreviation=false"

    yield "upper-number={0}".format(upperNumber)

    if (i > 0 and (words[i - 1] == 'De') and ((words[i] == 'Morgen'))):
        yield "is-ne=true"

    if (words[i] == 'Fra' or words[i] == 'Ita' or words[i] == 'Spa'):
        yield "is-ne=true"

    if (words[i].count('-') > 0):
        yield "has-dash=true"
    else:
        yield "has-dash=false"


    if (i > 0):
        yield "pos-previous={0}".format(poses[i - 1])
        yield "word-previous={0}".format(words[i - 1])

    if (i + 1 < len(poses)):
        yield "pos-next={0}".format(poses[i + 1])
        yield "word-next={0}".format(words[i + 1])

    if (i + 2 < len(poses)):
        yield "pos-next-next={0}".format(poses[i + 2])

    if (i > 1):
        yield "pos-previous-previous={0}".format(poses[i - 2])
        # yield "word-previous-previous={0}".format(words[i - 2])

    labels = data["labelled_words"].get(words[i], dict())
    labels = filter(lambda item: item[1] > MIN_LABEL_FREQUENCY, labels.items())

    if (i == 0):
        yield "in-beggining=true"
    else:
        yield "in-beggining=false"

    #if (i+1 == len(poses)):
        #yield "in-end=true"
    #else:
        #yield "in-end=false"

    if i > 0 and str.istitle(words[i][0]):
        yield "is-ne=true"

    if str.istitle(words[i][0]):
        yield "is-capital=true"
    else:
        yield "is-capital=false"

    for label in labels:
        yield "was-labelled-as={0}".format(label)


def isName(word):
    if not str.istitle(word[0]):
        return False
    word = word.upper()
    global person_names
    return word in person_names

def isSurname(word):
    if not str.istitle(word[0]):
        return False
    word = word.upper()
    global person_surnames
    return word in person_surnames

def wordInSentence(word, sentence):
    for sentence_word in sentence:
        if word == sentence_word[0]:
            return True
    return False
    
def hasDigits(word):
    for letter in word:
        if (str.isdigit(letter)):
            return True
    return False

def isDigits(word):
    for letter in word:
        if (not str.isdigit(letter)):
            return False
    return True

import string
def isPunctuation(word):
    for letter in word:
        if (letter not in string.punctuation):
            return False
    return True

def isURL(word):
    if (word.count("www.") > 0):
        return True
    return (word.count("http:") > 0)

def hasPunctuation(word):
    for letter in word:
        if (letter in string.punctuation):
            return True
    return False

def hasAlpha(word):
    for letter in word:
        if (str.isalpha(letter)):
            return True
    return False

# locClues = ['hoofdstad', 'regio', 'provincie', 'deelstaat']

def getWordFeatures(data, words, poses, i):

    if data["word_frequencies"].get(words[i], 0) >= MIN_WORD_FREQUENCY:
        yield "word-current={0}".format(words[i])

    if isName(words[i]):
        yield "is-name=true"

    if isSurname(words[i]):
        yield "is-surname=true"

    # global locClues
    # if words[i] in locClues:
    #     yield "is-locclue=true"
    # if i > 0 and isSurname(words[i - 1]) and isName(words[i]):
    #     yield "is-name-pair=true"
    # else:
    #     yield "is-name=false"

    # if i > 0 and isName(words[i - 1]) and isSurname(words[i]):
    #     yield "is-name-pair=true"

    yield "pos={0}".format(poses[i])
    # yield "ne-label={0}".format(ne_labels[i])

    if (len(words[i]) > 3):
        yield "word[-2:]={0}".format(words[i][-2:])

    if (len(words[i]) > 4):
        yield "word[-3:]={0}".format(words[i][-3:])

    if (len(words[i]) > 5):
        yield "word[-4:]={0}".format(words[i][-4:])

    if (len(words[i]) > 3):
        yield "word[:2]={0}".format(words[i][:2])

    if (len(words[i]) > 4):
        yield "word[:3]={0}".format(words[i][:3])

    if (len(words[i]) > 3):
        yield "word[:-2]={0}".format(words[i][:-2])

    if (len(words[i]) > 4):
        yield "word[:-3]={0}".format(words[i][:-3])

    upperNumber = upperLettersNumber(words[i])
    if (upperNumber == len(words[i])):
        yield "is-upper=true"
    # els:
    #     yield "is-upper=false"

    if (upperNumber == 0):
        yield "is-lower=true"

    if (upperLettersNumber(words[1:]) > 0):
        yield "has-internal-upper=true"

    if hasDigits(words[i]):
        yield "has-digits=true"

    if isPunctuation(words[i]):
        yield "is-punctuation=true"

    if isURL(words[i]):
        yield "is-url=true"

    if (data["single_quotes"]):
        yield "in-single-quotes=true"
        
    if (data["double_quotes"]):
        yield "in-double-quotes=true"

    if hasPunctuation(words[i]):
        yield "has-punctuation=true"

    if hasDigits(words[i]) and hasAlpha(words[i]):
        yield "has-digits-and-alpha=true"

    if isDigits(words[i]):
        yield "is-digits=true"
    # else:
    #     yield "has-internal-upper=false"

    # if (upperNumber * 2 >= len(words[i])):
    #     yield "is-abbreviation=true"
    # else:
    #     yield "is-abbreviation=false"

    yield "upper-number={0}".format(upperNumber)

    if (words[i] == 'Fra' or words[i] == 'Ita' or words[i] == 'Spa'):
        yield "is-location=true"

    if (words[i][0] != '-' and words[i][-1] != '-' and words[i].count('-') > 0):
        yield "has-dash=true"
    else:
        yield "has-dash=false"

    if (len(words[i]) <= 3):
        yield "small_len=true"

    # if (len(words[i]) >= 7):
    #     yield "long_len=true"
    # else:
    #     if (len(words[i]) <= 7):
    #         yield "medium_len=true"
    #yield "word-len={0}".format(len(words[i]))

    labels = data["labelled_words"].get(words[i], dict())
    labels = filter(lambda item: item[1] > MIN_LABEL_FREQUENCY, labels.items())

    if (i == 0):
        yield "in-beggining=true"
    else:
        yield "in-beggining=false"

    #if (i+1 == len(poses)):
        #yield "in-end=true"
    #else:
        #yield "in-end=false"


    if str.istitle(words[i][0]):
        yield "is-capital=true"
    else:
        yield "is-capital=false"

    for label in labels:
        yield "was-labelled-as={0}".format(label)

def compute_features(data, words, poses, i, previous_label):
    # Condition on previous label.
    #ne_labels = data["ne_labels"]

    # sentences = data["sentences"]
    # sentence_number = data["sentence_number"]

    # window_size = 10
    # lower_bound = max(sentence_number - window_size, 0)
    # upper_bound = min(sentence_number + window_size, len(sentences))

    # occurenceNumber = 0
    # for number in range(lower_bound, upper_bound):
    #     if (wordInSentence(words[i], sentences[number])):
    #         occurenceNumber = occurenceNumber + 1

    # yield "many-occurences={0}".format(occurenceNumber)
    # if occurenceNumber > 1:
    #     yield "many-occurences=true"
    # else:
    #     yield "many-occurences=false"


    if (words[i] == '"'):
        data["double_quotes"] = not data["double_quotes"]

    if (words[i] == "'"):
        data["single_quotes"] = not data["single_quotes"]

    if previous_label != "O":
        yield "label-previous={0}".format(previous_label) 

    for feature in getWordFeatures(data, words, poses, i):
        yield "current" + feature

    yield "pos={0}".format(poses[i])
    # yield "ne-label={0}".format(ne_labels[i])

    # if (i > 0 and (words[i - 1] == 'De') and ((words[i] == 'Morgen'))):
    #     yield "is-organization=true"

    if (i > 0):
        for feature in getWordFeatures(data, words, poses, i - 1):
            yield "prev-" + feature

    if (i > 1):
        for feature in getWordFeatures(data, words, poses, i - 2):
            yield "prev-prev-" + feature

    if (i + 1 < len(poses)):
        for feature in getWordFeatures(data, words, poses, i + 1):
            yield "next-" + feature

    if (i + 2 < len(poses)):
        for feature in getWordFeatures(data, words, poses, i + 2):
            yield "next-next-" + feature

def train_ne_binary_model(options, iterable):
    model = MaxentModel()
    data = {}

    data["feature_set"] = set()
    data["word_frequencies"] = defaultdict(long)
    # XXX(sandello): defaultdict(lambda: defaultdict(long)) would be
    # a better choice here (for |labelled_words|) but it could not be pickled.
    # C'est la vie.
    data["labelled_words"] = dict()

    print >>sys.stderr, "*** Training options are:"
    print >>sys.stderr, "   ", options

    print >>sys.stderr, "*** First pass: Computing statistics..."
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        for word, pos, label in sentence:
            data["word_frequencies"][word] += 1
            if label.startswith("B-") or label.startswith("I-"):
                if word not in data["labelled_words"]:
                    data["labelled_words"][word] = defaultdict(long)
                data["labelled_words"][word][label] += 1

    print >>sys.stderr, "*** Second pass: Collecting features..."
    model.begin_add_event()
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses, labels = map(list, zip(*sentence))
        for i in xrange(len(labels)):
            features = compute_ne_features(data, words, poses, i, labels[i - 1] if i >= 1 else "^")
            features = list(features)
            if labels[i].startswith("B-") or labels[i].startswith("I-"):
                model.add_event(features, "NE")
            else:
                model.add_event(features, "O")

            for feature in features:
                data["feature_set"].add(feature)
    model.end_add_event(options.cutoff)
    print >>sys.stderr, "*** Collected {0} features.".format(len(data["feature_set"]))

    print >>sys.stderr, "*** Training..."
    maxent.set_verbose(1)
    model.train(options.iterations, options.technique, options.gaussian)
    maxent.set_verbose(0)

    print >>sys.stderr, "*** Saving..."
    model.save(options.model + ".ne.binary.maxent")
    with open(options.model + ".ne.binary.data", "w") as handle:
        cPickle.dump(data, handle)

# |iterable| should yield sentences.
# |iterable| should support multiple passes.
def train_model(options, iterable):

    # train_ne_binary_model(options, iterable)
    # ne_labels = eval_ne_binary_model_train(options, iterable)

    model = MaxentModel()
    data = {}

    data["feature_set"] = set()
    data["word_frequencies"] = defaultdict(long)
    # XXX(sandello): defaultdict(lambda: defaultdict(long)) would be
    # a better choice here (for |labelled_words|) but it could not be pickled.
    # C'est la vie.
    data["labelled_words"] = dict()

    print >>sys.stderr, "*** Training options are:"
    print >>sys.stderr, "   ", options

    print >>sys.stderr, "*** First pass: Computing statistics..."
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)

        for word, pos, label in sentence:
            data["word_frequencies"][word] += 1
            if label.startswith("B-") or label.startswith("I-"):
                if word not in data["labelled_words"]:
                    data["labelled_words"][word] = defaultdict(long)
                data["labelled_words"][word][label] += 1

    print >>sys.stderr, "*** Second pass: Collecting features..."
    model.begin_add_event()

    data["sentences"] = iterable
    for n, sentence in enumerate(iterable):
        if (n % 1000) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses, labels = map(list, zip(*sentence))
        # sentence_ne_labels = ne_labels[n]
        # data["ne_labels"] = sentence_ne_labels
        data["sentence_number"] = n
        data["double_quotes"] = False
        data["single_quotes"] = False

        for i in xrange(len(labels)):
            features = compute_features(data, words, poses, i, labels[i - 1] if i >= 1 else "^")
            features = list(features)
            model.add_event(features, labels[i])
            for feature in features:
                data["feature_set"].add(feature)
    model.end_add_event(options.cutoff)
    print >>sys.stderr, "*** Collected {0} features.".format(len(data["feature_set"]))

    print >>sys.stderr, "*** Training..."
    maxent.set_verbose(1)
    model.train(options.iterations, options.technique, options.gaussian)
    maxent.set_verbose(0)

    print >>sys.stderr, "*** Saving..."
    model.save(options.model + ".maxent")
    with open(options.model + ".data", "w") as handle:
        cPickle.dump(data, handle)

# |iterable| should yield sentences.
def eval_model(options, iterable):
    model = MaxentModel()
    data = {}

    # ne_labels = eval_ne_binary_model(options, iterable)

    print >>sys.stderr, "*** Loading..."
    model.load(options.model + ".maxent")
    with open(options.model + ".data", "r") as handle:
        data = cPickle.load(handle)

    print >>sys.stderr, "*** Evaluating..."
    data["sentences"] = iterable
    for n, sentence in enumerate(iterable):
        if (n % 100) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses = map(list, zip(*sentence))
        # data["ne_labels"] = ne_labels[n]
        data["sentence_number"] = n
        data["double_quotes"] = False
        data["single_quotes"] = False

        labels = eval_model_sentence(options, data, model, words, poses)

        for word, pos, label in zip(words, poses, labels):
            print label
        print

# |iterable| should yield sentences.
def eval_ne_binary_model_train(options, iterable):
    model = MaxentModel()
    data = {}

    predicted_labels = []

    print >>sys.stderr, "*** Loading..."
    model.load(options.model + ".ne.binary.maxent")
    with open(options.model + ".ne.binary.data", "r") as handle:
        data = cPickle.load(handle)

    print >>sys.stderr, "*** Evaluating..."
    for n, sentence in enumerate(iterable):
        if (n % 100) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses, labels = map(list, zip(*sentence))
        ne_labels = eval_ne_binary_model_sentence(options, data, model, words, poses)
        predicted_labels += [ne_labels]

    return predicted_labels

def eval_ne_binary_model(options, iterable):
    model = MaxentModel()
    data = {}

    predicted_labels = []

    print >>sys.stderr, "*** Loading..."
    model.load(options.model + ".ne.binary.maxent")
    with open(options.model + ".ne.binary.data", "r") as handle:
        data = cPickle.load(handle)

    print >>sys.stderr, "*** Evaluating..."
    for n, sentence in enumerate(iterable):
        if (n % 100) == 0:
            print >>sys.stderr, "   {0:6d} sentences...".format(n)
        words, poses = map(list, zip(*sentence))
        labels = eval_ne_binary_model_sentence(options, data, model, words, poses)
        predicted_labels += [labels]

    return predicted_labels

import random
def split_dataset(options, iterable):
    sentences = list(iterable)
    random.shuffle(sentences)
    n = len(sentences)
    #print(options.splitProportion);
    testSize = int(options.splitProportion * n)
    trainSize = n - testSize
    
    train = sentences[:trainSize]
    test = sentences[trainSize:]
    # options.filename + '.test'
    trainFile = open(options.filename + '.train', 'w')
    for n, sentence in enumerate(test):
        #print(sentence)
        for i in range(0, len(sentence)):
            trainFile.writelines(" ".join(sentence[i]) + '\n')
        trainFile.writelines('\n')

    trainFile.close();

    testFile = open(options.filename + '.test', 'w')
    testFileAns = open(options.filename + '.test.ans', 'w')

    for n, sentence in enumerate(train):
        words, poses, labels = map(list, zip(*sentence))

        for i in range(0, len(labels)):
            testFile.writelines(words[i] + " " + poses[i] + '\n')
            testFileAns.writelines(labels[i] + '\n')
        testFile.writelines('\n');
        testFileAns.writelines('\n')

    testFile.close();
    testFileAns.close();


# This is a helper method for |eval_model_sentence| and, actually,
# an implementation of Viterbi algorithm.
def eval_model_sentence(options, data, model, words, poses):
    viterbi_layers = [ None for i in xrange(len(words)) ]
    viterbi_backpointers = [ None for i in xrange(len(words) + 1) ]

    # Compute first layer directly.
    viterbi_layers[0] = model.eval_all(list(compute_features(data, words, poses, 0, "^")))
    viterbi_layers[0] = dict( (k, math.log(v)) for k, v in viterbi_layers[0] )
    viterbi_backpointers[0] = dict( (k, None) for k, v in viterbi_layers[0].iteritems() )

    # Compute intermediate layers.
    for i in xrange(1, len(words)):
        viterbi_layers[i] = defaultdict(lambda: float("-inf"))
        viterbi_backpointers[i] = defaultdict(lambda: None)
        for prev_label, prev_logprob in viterbi_layers[i - 1].iteritems():
            features = compute_features(data, words, poses, i, prev_label)
            features = list(features)
            for label, prob in model.eval_all(features):
                logprob = math.log(prob)
                if prev_logprob + logprob > viterbi_layers[i][label]:
                    viterbi_layers[i][label] = prev_logprob + logprob
                    viterbi_backpointers[i][label] = prev_label

    # Most probable endpoint.
    max_logprob = float("-inf")
    max_label = None
    for label, logprob in viterbi_layers[len(words) - 1].iteritems():
        if logprob > max_logprob:
            max_logprob = logprob
            max_label = label

    # Most probable sequence.
    path = []
    label = max_label
    for i in reversed(xrange(len(words))):
        path.insert(0, label)
        label = viterbi_backpointers[i][label]

    return path


def eval_ne_binary_model_sentence(options, data, model, words, poses):
    viterbi_layers = [ None for i in xrange(len(words)) ]
    viterbi_backpointers = [ None for i in xrange(len(words) + 1) ]

    # Compute first layer directly.
    viterbi_layers[0] = model.eval_all(list(compute_ne_features(data, words, poses, 0, "^")))
    viterbi_layers[0] = dict( (k, math.log(v)) for k, v in viterbi_layers[0] )
    viterbi_backpointers[0] = dict( (k, None) for k, v in viterbi_layers[0].iteritems() )

    # Compute intermediate layers.
    for i in xrange(1, len(words)):
        viterbi_layers[i] = defaultdict(lambda: float("-inf"))
        viterbi_backpointers[i] = defaultdict(lambda: None)
        for prev_label, prev_logprob in viterbi_layers[i - 1].iteritems():
            features = compute_ne_features(data, words, poses, i, prev_label)
            features = list(features)
            for label, prob in model.eval_all(features):
                logprob = math.log(prob)
                if prev_logprob + logprob > viterbi_layers[i][label]:
                    viterbi_layers[i][label] = prev_logprob + logprob
                    viterbi_backpointers[i][label] = prev_label

    # Most probable endpoint.
    max_logprob = float("-inf")
    max_label = None
    for label, logprob in viterbi_layers[len(words) - 1].iteritems():
        if logprob > max_logprob:
            max_logprob = logprob
            max_label = label

    # Most probable sequence.
    path = []
    label = max_label
    for i in reversed(xrange(len(words))):
        path.insert(0, label)
        label = viterbi_backpointers[i][label]

    return path

################################################################################

def main():
    parser = OptionParser("A sample MEMM model for NER")
    parser.add_option("-T", "--train", action="store_true", dest="train",
        help="Do the training, if specified; do the evaluation otherwise")

    parser.add_option("-G", "--genSplit", action="store_true", dest="split",
        help="Split dataset")

    parser.add_option("-S", "--splitProportion", type="float", dest="splitProportion",
        help="set split proportion", metavar=0.3)

    parser.add_option("-f", "--file", type="string", dest="filename",
        metavar="FILE", help="File with the training data")
    parser.add_option("-m", "--model", type="string", dest="model",
        metavar="FILE", help="File with the model")
    parser.add_option("-c", "--cutoff", type="int", default=5, dest="cutoff",
        metavar="C", help="Event frequency cutoff during training")
    parser.add_option("-i", "--iterations", type="int", default=100, dest="iterations",
        metavar="N", help="Number of training iterations")
    parser.add_option("-g", "--gaussian", type="float", default=0.0, dest="gaussian",
        metavar="G", help="Gaussian smoothing penalty (sigma)")
    parser.add_option("-t", "--technique", type="string", default="gis", dest="technique",
        metavar="T", help="Training algorithm (either 'gis' or 'lbfgs')")
    (options, args) = parser.parse_args()

    if not options.filename:
        parser.print_help()
        sys.exit(1)

    read_names('./dutch.names')
    read_surnames('./dutch.surnames')

    with open(options.filename, "r") as handle:
        data = list(read_sentences(handle))

    if options.split:
        print >>sys.stderr, "*** Splitting dataset..."
        split_dataset(options, data);
    else:
      if options.train:
          print >>sys.stderr, "*** Training model..."
          train_model(options, data)
      else:
          print >>sys.stderr, "*** Evaluating model..."
          eval_model(options, data)

    print >>sys.stderr, "*** Done!"

if __name__ == "__main__":
    main()

