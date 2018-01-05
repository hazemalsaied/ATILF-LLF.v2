import logging
import os
import sys

from corpus import Corpus
from evaluation import Evaluation
from oracles import EmbeddingOracle
from param import FeatParams, XPParams, Paths
from parsers import Parser


def identify():
    constantConfigFolder = Paths.configsFolder
    for configFile in os.listdir(Paths.configsFolder):
        if not configFile.endswith('.json'):
            continue
        corpus = Corpus(configFile[:2])
        FeatParams(os.path.join(constantConfigFolder, configFile), corpus=corpus)
        if XPParams.useCrossValidation:
            scores = [0] * 12
            testRange, trainRange = corpus.getRangs()
            for x in range(len(testRange)):
                logging.warn('Iteration no.' + str(x + 1))
                XPParams.currentIteration = x
                Paths.iterationPath = os.path.join(Paths.langResultFolder, str(x + 1))
                evalScores = identifyCorpus(corpus)
                for i in range(6):
                    scores[i] += evalScores[i]
                createMWEFiles(corpus, configFile[:2], x)
            for i in range(len(scores)):
                scores[i] /= float(len(testRange))
            logging.warn(' F-Score: ' + str(scores[0]))
        else:
            identifyCorpus(corpus)
            createMWEFiles(corpus, configFile[:2])


def identifyCorpus(corpus):
    corpus.update()
    clf = EmbeddingOracle.train(corpus)
    Parser.parse(corpus, clf)
    scores = Evaluation.evaluate(corpus)
    return scores


def createMWEFiles(corpus, lang, x=-1):
    folder = '../Results/MWEFiles'
    if XPParams.useCrossValidation:
        folder += '/CV/' + lang
    else:
        folder += '/testSet/'
    if XPParams.useCrossValidation and not os.path.exists(folder):
        os.makedirs(folder)
    if x == -1:
        x = ''
    mwePath = os.path.join(folder, lang + str(x) + '.txt')
    with open( mwePath, 'w') as f:
        logging.warn('MWE file is being written to {0}'.format(mwePath))
        f.write(str(corpus))
    goldenPath = os.path.join(folder, lang + str(x) + '.gold.txt')
    with open(goldenPath, 'w') as f:
        logging.warn(' Golden MWE file is being written to {0}'.format(goldenPath))
        f.write(corpus.getGoldenMWEFile())


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)

# XPParams.realExper = True
XPParams.useCrossValidation = True
identify()
