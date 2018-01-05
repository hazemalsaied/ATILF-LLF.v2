import datetime
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from features import Extractor
from transitions import Reduce, BlackMerge, EmbeddingTransition, MergeAsMWT
from transitions import Shift


class EmbeddingOracle:
    @staticmethod
    def train(corpus):
        time = datetime.datetime.now()
        logging.info('Static Embedding Oracle')
        Y, X_dic = EmbeddingOracle.parseCorpus(corpus.trainingSents, EmbeddingOracle)
        vec = DictVectorizer()
        X = vec.fit_transform(X_dic)
        clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
        clf.fit(X, Y)
        logging.info('Traingin Time: ' + str(int((datetime.datetime.now() - time).seconds / 60.)))
        return clf, vec

    @staticmethod
    def parseCorpus(sents, cls):
        labels, features = [], []
        for sent in sents:
            # Parse the sentence
            trainingInfo = cls.parseSentence(sent, cls)
            if trainingInfo is not None:
                labels.extend(trainingInfo[0])
                features.extend(trainingInfo[1])

        return labels, features

    @staticmethod
    def parseSentence(sent, cls):
        sent.initialTransition = EmbeddingTransition(isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            transition = cls.getNextTransition(transition, sent)
        labels, features = Extractor.extract(sent)
        return labels, features

    @staticmethod
    def getNextTransition(parent, sent):
        newTransition = MergeAsMWT.check(parent)
        if newTransition is not None:
            return newTransition

        newTransition = BlackMerge.check(parent)
        if newTransition is not None:
            return newTransition

        # Check for VMWE complete
        newTransition = Reduce.check(parent)
        if newTransition is not None:
            return newTransition

        # Apply the default transition: SHIFT
        shift = Shift(sent=sent)
        shift.apply(parent, sent)
        return shift
