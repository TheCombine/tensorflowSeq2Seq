from time import strftime, localtime
from os import listdir
import numpy as np

def LoadVocabulary():
    fileNames = listdir('savesVocabulary/')
    if len(fileNames) == 0:
        return None
    recentFileName = None
    recentCreationDate = -1
    for fileName in fileNames:
        try:
            creationDate = int(''.join([c for c in fileName if c.isdigit()]))
        except:
            continue
        if creationDate > recentCreationDate:
            recentFileName = fileName
            recentCreationDate = creationDate
    if recentFileName == None:
        return None
    print('Recent vocabulary save:', recentFileName)
    with open('savesVocabulary/%s' % recentFileName, 'r') as file:
        contents = file.read()
    vocabulary = Vocabulary()
    StrTokenObj, StrTokenSrc = contents.split('<<<VOCABULARY SEPARATOR>>>')

    rows = filter(None, StrTokenObj.split('\n'))
    for row in rows:
        i, token = row.split()
        if token not in vocabulary.token_to_index_obj:
            i = int(i)
            vocabulary.index_to_token_obj.insert(i, token)
            vocabulary.token_to_index_obj[token] = i

    rows = filter(None, StrTokenSrc.split('\n'))
    for row in rows:
        i, token = row.split()
        if token not in vocabulary.token_to_index_src:
            i = int(i)
            vocabulary.index_to_token_src.insert(i, token)
            vocabulary.token_to_index_src[token] = i

    return vocabulary

class Vocabulary(object):
    def SaveVocabulary(self):
        with open('savesVocabulary/%s.txt' % strftime('%Y-%m-%d %H;%M;%S', localtime()), 'w') as file:
            for i in range(len(self.index_to_token_obj)):
                print('%s %s' % (i, self.index_to_token_obj[i]), file=file)
            print('<<<VOCABULARY SEPARATOR>>>', file=file)
            for i in range(len(self.index_to_token_src)):
                print('%s %s' % (i, self.index_to_token_src[i]), file=file)

    def __init__(self):
        self.token_to_index_obj = {}
        self.index_to_token_obj = []
        self.token_to_index_src = {}
        self.index_to_token_src = []

        self.special_tokens = ['<unk>', '<str>', '<end>']
        for i in range(len(self.special_tokens)):
            self.token_to_index_obj[self.special_tokens[i]] = i
            self.index_to_token_obj.insert(i, self.special_tokens[i])
            self.token_to_index_src[self.special_tokens[i]] = i
            self.index_to_token_src.insert(i, self.special_tokens[i])

    def _applyElemWise(self, List, callback):
        _List = []
        for item in List:
            if isinstance(item, (list, tuple)):
                _List.append(self._applyElemWise(item, callback))
            else:
                _List.append(callback(item))
        return _List 

    def _feedToken(self, Token, token_to_index, index_to_token):
        if Token not in token_to_index:
            i = len(index_to_token)
            token_to_index[Token] = i
            index_to_token.insert(i, Token)

    def FeedTokensObj(self, TokensObj):
        return self._applyElemWise(TokensObj, lambda Token: self._feedToken(Token, self.token_to_index_obj, self.index_to_token_obj))

    def FeedTokensSrc(self, TokensSrc):
        return self._applyElemWise(TokensSrc, lambda Token: self._feedToken(Token, self.token_to_index_src, self.index_to_token_src))

    def TokensToIndicesObj(self, TokensObj):
        return self._applyElemWise(TokensObj, lambda Token: self.token_to_index_obj[Token])

    def TokensToIndicesSrc(self, TokensSrc):
        return self._applyElemWise(TokensSrc, lambda Token: self.token_to_index_src[Token])

    def IndicesToTokensObj(self, IndicesObj):
        return self._applyElemWise(IndicesObj, lambda Index: self.index_to_token_obj[Index])

    def IndicesToTokensSrc(self, IndicesSrc):
        return self._applyElemWise(IndicesSrc, lambda Index: self.index_to_token_src[Index])
