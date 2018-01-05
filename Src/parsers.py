from features import Extractor
from param import XPParams
from transTypes import TransitionType
from transitions import EmbeddingTransition, Transition


class Parser:
    @staticmethod
    def parse(corpus, clf):
        if XPParams.useCrossValidation:
            corpus.initializeSents(training=False)
        for sent in corpus.testingSents:
            sent.initialTransition = EmbeddingTransition(None, isInitial=True, sent=sent)
            transition = sent.initialTransition
            feats, labels = [], []
            while not transition.configuration.isTerminalConf():
                newTransition = Parser.getNextTransition(transition, sent, clf[0], clf[1], feats)
                if newTransition:
                    newTransition.apply(transition, sent, parse=True)
                    labels.append(newTransition.type.value)
                    transition = newTransition
            sent.featuresInfo = [labels, feats]

    @staticmethod
    def getNextTransition(transition, sent, classifier, dictVectorizer, feats):

        legalTansDic = transition.getLegalTransDic()
        if len(legalTansDic) == 1:
            return Transition.initialize(legalTansDic.keys()[0], sent)
        featDic = Extractor.getFeatures(transition, sent)
        if not isinstance(featDic, list):
            featDic = [featDic]
        transTypeValue = classifier.predict(dictVectorizer.transform(featDic))[0]
        transType = TransitionType.getType(transTypeValue)
        if transType in legalTansDic:
            return legalTansDic[transType]
        if len(legalTansDic):
            return Transition.initialize(legalTansDic.keys()[0], sent)
        raise
