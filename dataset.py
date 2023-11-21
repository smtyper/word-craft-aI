import os
import pandas as pd
import numpy as np

from keras.utils import to_categorical
from yo_fluq import Query


def getNounsWords(count = 1000000, minLen = 2, maxLen = 8) -> list[str]:
    nounsFilePath = os.path.join("data", "nouns.csv")

    def hasDuplicateChars(string):
        for i in range(len(string) - 1):
            if string[i] == string[i+1]:
                return True
        return False

    df = pd.read_csv(nounsFilePath, sep = "\t")
    nouns = [row.bare.lower() for _, row in df.iterrows()
             if len(row.bare) >= minLen and len(row.bare) <= maxLen and not hasDuplicateChars(row.bare)][:count]

    return nouns


def splitWords(words: list[str], wordsInGroupCount: int) -> list[tuple[str, list[str]]]:
    def segmentize(group, segmentLength: int):
        group = Query.en(group).to_list()
        segments = [group[word:word + segmentLength] for word in range(0, len(group), segmentLength)]

        return segments

    lengths = [i for i in range(3, 4)]

    splittedWords: dict = (Query
            .en(lengths)
            .select_many(lambda length: (Query
                .en(words)
                .where(lambda word: len(word) > length)
                .group_by(lambda word: word[:length])
                .select_many(lambda group: Query
                             .en(segmentize(group, wordsInGroupCount))
                             .where(lambda segment: len(segment) == wordsInGroupCount)
                             .select(lambda segment: (group.key, segment)))))
            .to_list())

    return splittedWords


def getSetsIndices(totalCount: int, training_set_percent: float = 0.85):
    permutation = np.random.permutation(totalCount)
    trainUpTo = int(totalCount * training_set_percent)

    trainIndices = permutation[:trainUpTo]
    testIndices = permutation[trainUpTo:]

    return trainIndices, testIndices


def toVectors(word: str, outputLen: int, alphabetLetters: str):
    arrays = [np.array(to_categorical(alphabetLetters.find(word[index] if index < len(word) else " "),
                                      len(alphabetLetters)))
                                      for index in range(outputLen)]

    return arrays


def fitDataset(splittedWords: list[tuple[str, list[str]]],
               alphabetLetters: str,
               inputWordLen: int,
               outputWordCount: int,
               outputWordLen: int,
               batchSize = 64, 
               epochs = 5):
    for _ in range(epochs):
        inputs, outputs = [], []

        for wordStart, wordEnds in splittedWords:
            if len(inputs) == batchSize:
                yield np.stack(inputs), np.stack(outputs)
                inputs, outputs = [], []

            inputWord = toVectors(wordStart, inputWordLen, alphabetLetters)
            outputWords = [np.stack(toVectors(wordEnds[wordIndex] if wordIndex < len(wordEnds) else "", outputWordLen,
                                              alphabetLetters))for wordIndex in range(outputWordCount)]

            # temp = [wordEnds[wordIndex] if wordIndex < len(wordEnds) else "" for wordIndex in range(outputWordCount)]

            # print(max(temp, key = lambda word: len(word)))

            inputs.append(inputWord)
            outputs.append(outputWords)

        if len(inputs) != 0:
            yield inputs, outputs
