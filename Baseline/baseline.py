import copy
import logging
import os
import sys

from Src.corpus import Corpus, VMWE
from Src.evaluation import Evaluation
from Src.param import XPParams

corporaPath = '../sharedtask'
resultPath = '../Baseline/Results/testDataSet/{0}.txt'
cvResultFileName = '{0}.txt'
cvGoldFileName = '{0}.gold.txt'
corpusLangPath = '../Baseline/Results/CV/Corpus/{0}/'
trainDataSetLangPath = '../Baseline/Results/CV/TrainDataSet/{0}/'


def identifyCV(onTrainDataSet=False):
    for subdir, dirs, files in os.walk(corporaPath):
        for lang in dirs:
            corpus = Corpus(lang)
            if not onTrainDataSet:
                corpus.trainDataSet = corpus.trainDataSet + corpus.testDataSet
            foldNum = 5
            for foldIdx in range(foldNum):
                corpus.divideCorpus(foldIdx)
                print len(corpus.trainingSents)
                print len(corpus.testingSents)
                mweDictionary = corpus.getMWEDictionaryWithWindows()
                recognizeMWEs(corpus, mweDictionary)
                Evaluation.evaluate(corpus)
                langPath = corpusLangPath.format(
                    lang) if not onTrainDataSet else trainDataSetLangPath.format(lang)
                if not os.path.exists(langPath):
                    os.makedirs(langPath)
                with open(os.path.join(langPath, cvGoldFileName).format(lang + str(foldIdx + 1)), 'w') as f:
                    f.write(corpus.getGoldenMWEFile())
                with open(os.path.join(langPath, cvResultFileName.format(lang + str(foldIdx + 1))), 'w') as f:
                    f.write(str(corpus))


def identify():
    for subdir, dirs, files in os.walk(corporaPath):
        for lang in dirs:
            corpus = Corpus(lang)
            mweDictionary = corpus.getMWEDictionaryWithWindows()
            recognizeMWEs(corpus, mweDictionary)
            Evaluation.evaluate(corpus)
            with open(resultPath.format(lang), 'w') as f:
                f.write(str(corpus))


def recognizeMWEs(corpus, mweDictionary):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        mweIdx = 1
        for entry in mweDictionary.keys():
            idxs = isInSentAndCloseEnough(entry, sent, mweDictionary[entry].split(';'))
            if idxs:
                for idx in idxs:
                    # idxs = idxs[0]
                    vmwe = VMWE(mweIdx)
                    for id in idx:
                        vmwe.tokens.append(sent.tokens[id])
                    sent.identifiedVMWEs.append(vmwe)
                    mweIdx += 1


def generateGoldenFiles():
    for subdir, dirs, files in os.walk(corporaPath):
        for lang in dirs:
            if lang == 'CS':
                continue
            corpus = Corpus(lang)
            foldNum = 5
            for foldIdx in range(foldNum):
                corpus.divideCorpus(foldIdx)
                langPath = '../Baseline/CVResults/{0}/'.format(lang)
                if not os.path.exists(langPath):
                    os.makedirs(langPath)

                with open(cvGoldFileName.format(lang, lang + str(foldIdx + 1)), 'w') as f:
                    f.write(corpus.getGoldenMWEFile())


def getLemmaList(tokens):
    lemmaList = []
    for token in tokens:
        lemmaList.append(token.getLemma())
    return lemmaList


def tokenize(entry):
    tokens = []
    for item in entry.split(' '):
        if item:
            tokens.append(item)
    return tokens


def isInSentAndCloseEnough(mweEntry, sent, windows, tolerance=0, ordered=True):
    sentLemmaList = getLemmaList(sent.tokens)
    mweLemmaList = tokenize(mweEntry)
    for lemma in mweLemmaList:
        if lemma not in sentLemmaList:
            return False
    positionsOfMWELemmas = []
    for lemma in mweLemmaList:
        positionsOfMWELemmas.append([i for i, x in enumerate(sentLemmaList) if x == lemma])
    if len(mweLemmaList) == 1:
        return positionsOfMWELemmas
    permutations, step = [], 0
    for i in range(len(positionsOfMWELemmas)):
        if not permutations:
            for item1 in positionsOfMWELemmas[i]:
                if item1:
                    permutations.append([item1])
        else:
            for item1 in positionsOfMWELemmas[i]:
                for perm in permutations:
                    if len(perm) == step:
                        permCopy = copy.deepcopy(perm)
                        permCopy.append(item1)
                        permutations.append(permCopy)
        step += 1
    acceptedPermutations = []
    for perm in permutations:
        if len(perm) == len(windows) + 1:
            if ordered:
                if sorted(perm) != perm:
                    continue
            accepted = True
            for i in range(len(perm) - 1):
                distance = perm[i + 1] - perm[i]
                if not ordered and distance < 0:
                    distance *= -1
                if distance > int(windows[i]) + tolerance:
                    accepted = False
            if accepted and acceptedPermutations:
                for elem in perm:
                    for accPerm in acceptedPermutations:
                        for elem2 in accPerm:
                            if elem == elem2:
                                accepted = False
            if accepted:
                acceptedPermutations.append(perm)
    return acceptedPermutations


def entryInSent(sent, entry):
    lemmatizedText = ' '
    for token in sent.tokens:
        lemmatizedText += token.getLemma() + ' '
    entryComps = entry.split(' ')
    isInSent = True
    expTokens = []
    for entryComp in entryComps:
        if ' ' + entryComp + ' ' in lemmatizedText:
            for token in sent.tokens:
                if token.getLemma() == entryComp:
                    expTokens.append(token)
                    break
        else:
            isInSent = False
            break

    return isInSent, expTokens


def fromTxt2CSV(resultTXT, resultCSV, crossValidation=False):
    fScores = []
    suffix = '  * F ='
    with open(resultTXT, 'r') as evalFile:
        for line in evalFile:
            if line.startswith(suffix):
                fScores.append(float(line[len(suffix):-1].strip()))
    mweBased, tokenBased, res, idx = 0, 0, '', 1
    for score in fScores:
        if idx % 2 == 0:
            tokenBased += score
        else:
            mweBased += score
        if crossValidation and idx and idx % 10 == 0:
            mweBased = float(mweBased / 5)
            tokenBased = float(tokenBased / 5)
            res += '{0},{1}\n'.format(mweBased, tokenBased)
            mweBased, tokenBased = 0, 0
        elif not crossValidation and idx and idx % 2 == 0:
            res += '{0},{1}\n'.format(mweBased, tokenBased)
            mweBased, tokenBased = 0, 0
        idx += 1
    with open(resultCSV, 'w') as resF:
        resF.write(res)


resultTrainSetTXT = '../Baseline/Results/results.txt'
resultTrainSetCSV = '../Baseline/Results/results.csv'

reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
XPParams.baseline = True
XPParams.realExper = True
# generateGoldenFiles()
# identify()
# # print 'CROSS VALIDATION'
# XPParams.realExper= False
# identifyCV(onTrainDataSet=True)
# fromTxt2CSV(resultTrainSetTXT, resultTrainSetCSV)
fromTxt2CSV(resultTrainSetTXT, resultTrainSetCSV, crossValidation=True)

# fromTxt2CSV(resultCorpusTXT, resultCorpusCSV)
