import re

o1test = 'data/artificial/ds_O1_test_5011.txt'
o2test = 'data/artificial/ds_O2_test_4977.txt'
o3test = 'data/artificial/ds_O3_test_5159.txt'
o1train = 'data/artificial/ds_O1_train_39977.txt'
o2train = 'data/artificial/ds_O2_train_39945.txt'
o3train = 'data/artificial/ds_O3_train_39811.txt'
o1valid = 'data/artificial/ds_O1_valid_5012.txt'
o2valid = 'data/artificial/ds_O2_valid_5078.txt'
o3valid = 'data/artificial/ds_O3_valid_5030.txt'

class DataEngine(object):
    def LoadData(self, path):
        """out: SeqSeqTokenObj, SeqSeqTokenSrc, SeqStringAli"""
        #read all file contents
        file = open(path)
        contents = file.read()
        file.close()
        #split contents into obj, src, ali
        TokensObj = []
        TokensSrc = []
        for Function in contents.split('<<<sep_out_sample>>>\n'):
            StringObj, StringSrc, _ = Function.split('<<<sep_in_sample>>>\n')
            SeqLineObj = self._tokenize(StringObj, '\n|%[a-z0-9]+|\$|,|\(|\)|\-|:|<|>|\+|[a-zA-Z0-9_]+') # [self._tokenize(LineObj, '%[a-z0-9]+|\$|,|\(|\)|\-|:|<|>|\+|[a-zA-Z0-9_]+') for LineObj in StringObj.split('\n') if LineObj]
            SeqLineSrc = self._tokenize(StringSrc, '\n|if|else|switch|case|default|break|int|float|char|double|long|for|while|do|void|goto|auto|signed|const|extern|register|unsigned|return|continue|enum|sizeof|struct|typedef|union|volatile|\+|\-|/|\*|&|\||\^|~|\?|\(|\)|\[|\]|\{|\}|<|>|=|!|,|\'|"|;|#|\\\|[a-zA-Z0-9_]+') # [self._tokenize(LineSrc, 'if|else|switch|case|default|break|int|float|char|double|long|for|while|do|void|goto|auto|signed|const|extern|register|unsigned|return|continue|enum|sizeof|struct|typedef|union|volatile|\+|\-|/|\*|&|\||\^|~|\?|\(|\)|\[|\]|\{|\}|<|>|=|!|,|\'|"|;|#|\\\|[a-zA-Z0-9_]+') for LineSrc in StringSrc.split('\n') if LineSrc]
            TokensObj.append(SeqLineObj)
            TokensSrc.append(SeqLineSrc)
        return TokensObj, TokensSrc

    def _tokenize(self, str, Pattern):
        """Called by LoadData"""
        SeqToken = re.findall(Pattern, str)
        #split everything thats not a variable name (re [a-zA-Z0-9_]+)
        for Token in SeqToken:
            if Token[0].isdigit():
                TokenIndex = SeqToken.index(Token)
                SeqToken = SeqToken[0:TokenIndex] + list(Token) + SeqToken[TokenIndex + 1:]
            elif Token == '\n':
                TokenIndex = SeqToken.index(Token)
                SeqToken = SeqToken[0:TokenIndex] + ['<nl>'] + SeqToken[TokenIndex + 1:]
        return SeqToken