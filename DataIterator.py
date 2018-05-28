import numpy as np

class DataIterator(object):
    def __init__(self, IndicesObj, IndicesSrc):
        self.IndicesObj = IndicesObj
        self.IndicesSrc = IndicesSrc
        self.size = len(IndicesObj)
        self.pointer = 0;

    def GetNextBatchNLP(self, n: int):
        n = min(n, len(self.IndicesObj) - self.pointer)
        if n <= 0:
            return None
        X_len = [len(fn) + 1 for fn in self.IndicesObj[self.pointer:self.pointer+n]]
        Y_len = [len(fn) + 1 for fn in self.IndicesSrc[self.pointer:self.pointer+n]]
        X = np.zeros([n, max(X_len)], np.int32)
        for i, x in enumerate(self.IndicesObj[self.pointer:self.pointer+n]):
            X[i, 0] = 1 #start token
            X[i, 1:X_len[i]] = x
        Y = np.zeros([n, max(Y_len)], np.int32)
        Y_targets = np.zeros([n, max(Y_len)], np.int32)
        for i, y in enumerate(self.IndicesSrc[self.pointer:self.pointer+n]):
            Y[i, 0] = 1 #start token
            Y[i, 1:Y_len[i]] = y
            Y_targets[i, :Y_len[i] - 1] = y
            Y_targets[i, Y_len[i] - 1] = 2 #end token
        self.pointer += n
        return X, X_len, Y, Y_len, Y_targets

    def getNextBatch(self):
        if self.size > self.pointer:
            padded_lines_obj, line_lenghts_obj = self._pad(self.pointer, self.IndicesObj)
            padded_lines_src, line_lenghts_src = self._pad(self.pointer, self.IndicesSrc)
            self.pointer += 1
            return padded_lines_obj, line_lenghts_obj, padded_lines_src, line_lenghts_src
        return None

    def _pad(self, index, Indices):
        lines = Indices[index]
        line_lenghts = [len(line) for line in lines]
        padded_lines = np.zeros([len(lines), max(line_lenghts)], dtype=np.int32)
        for i, padded_line in enumerate(padded_lines):
            padded_line[:line_lenghts[i]] = lines[i]
        return padded_lines, line_lenghts

    def getNextBatchV2(self):
        if self.size > self.pointer:
            lines_obj = self.IndicesObj[self.pointer]
            line_lengths_obj = [len(line_obj) for line_obj in lines_obj]
            lines_src = self.IndicesSrc[self.pointer]
            line_lengths_src = [len(line_src) for line_src in lines_src]

            rows = max(len(lines_obj), len(lines_src))
            cols = max(max(line_lengths_obj), max(line_lengths_src))

            padded_lines_obj = np.zeros(shape=[rows, cols], dtype=np.int32)
            for i, line_obj in enumerate(lines_obj):
                padded_lines_obj[i, :line_lengths_obj[i]] = lines_obj[i]
            padded_line_lengths_obj = np.zeros(shape=[rows], dtype=np.int32)
            for i, line_length_obj in enumerate(line_lengths_obj):
                padded_line_lengths_obj[i] = line_length_obj

            padded_lines_src = np.zeros(shape=[rows, cols], dtype=np.int32)
            for i, line_src in enumerate(lines_src):
                padded_lines_src[i, :line_lengths_src[i]] = lines_src[i]
            padded_line_lengths_src = np.zeros(shape=[rows], dtype=np.int32)
            for i, line_length_src in enumerate(line_lengths_src):
                padded_line_lengths_src[i] = line_length_src

            self.pointer += 1
            return padded_lines_obj, line_lengths_obj, padded_lines_src, line_lengths_src
        return None


