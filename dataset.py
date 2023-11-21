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


def splitWords(words: list[str]) -> dict[str, list[str]]:
    lengths = [i for i in range(3, 7)]

    splittedWords: dict = (Query
            .en(lengths)
            .select_many(lambda length: (Query
                .en(words)
                .where(lambda word: len(word) > length)
                .group_by(lambda word: word[:length])))
            .to_dictionary(lambda group: group.key,
                           lambda group: (Query
                                          .en(group)
                                          .select(lambda word: word[len(group.key):])
                                        #   .group_by(lambda word: len(word))
                                        #   .select_many(lambda lenGroup: Query.en(lenGroup).take(lenGroupWordCount))
                                          .order_by_descending(lambda word: len(word))
                                        #   .then_by(lambda word: word)
                                          .to_list())))

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


def fitDataset(splittedWords: dict[str, list[str]],
               indices: np.ndarray,
               alphabetLetters: str,
               inputWordLen: int,
               outputWordCount: int,
               outputWordLen: int,
               batchSize = 64, 
               epochs = 5):
    for _ in range(epochs):
        setWords = splittedWords.items()
        inputs, outputs = [], []

        for wordStart, wordEnds in setWords:
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
