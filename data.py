import codecs

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2 # count SOS and EOS

    def addWord(self, word):
        for char in list(word):
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def inVocab(self, word):
        for char in list(word):
            if char not in self.char2index:
                return False
        return True


def filterPair(p, max_length):
    return len(list(p[0])) < max_length and len(list(p[1])) < max_length 


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def readLangs(filepath, lang1, lang2, reverse=False, max_length = 20):
    print("Reading lines in "+filepath)

    f = codecs.open(filepath, encoding='utf-8')
    lines = f.readlines()
    f.close()

    #split every line into pair of words
    pairs = filterPairs([l.strip().split('\t') for l in lines],max_length)
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareTrainData(filepath, lang1, lang2, reverse=False, max_length = 20):
    input_lang, output_lang, pairs = readLangs(filepath, lang1, lang2, reverse, max_length)
    print("Read %s word pairs" % len(pairs))
    print("Vocabulary statistics")
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

def prepareTestData(filepath, inputlang, outputlang, reverse=False, max_length = 20):
    f = codecs.open(filepath, encoding='utf-8')
    lines = f.readlines()
    f.close()
    #split every line into pair of words
    pairs = filterPairs([l.strip().split('\t') for l in lines],max_length)
    print("Read %s word pairs" % len(pairs))
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
    invocab_pairs = [p for p in pairs if inputlang.inVocab(p[0]) and outputlang.inVocab(p[1])]

    #    invocab_pairs = [p for p in pairs if outputlang.inVocab(p[0]) and inputlang.inVocab(p[1])]
    print("Keeping %s word pairs for which all characters are in vocabulary" % len(invocab_pairs))
    return invocab_pairs

