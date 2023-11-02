import os
import pandas as pd
import numpy as np

from keras.utils import to_categorical


def getNounsWords(count = 1000000, minLen = 2, maxLen = 15) -> list[str]:
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


def fitDataset(words: list[str],
               indices: np.ndarray,
               alphabetLetters: str,
               inputWordLen: int,
               outputWordLen: int,
               batchSize = 64, 
               epochs = 5):
    for _ in range(epochs):
        setWords = [words[index] for index in indices]
        inputs, outputs = [], []

        for word in setWords:
            if len(inputs) == batchSize:
                yield np.stack(inputs), np.stack(outputs)
                inputs, outputs = [], []

            inputs.append(toVectors(word, inputWordLen, alphabetLetters))
            outputs.append(toVectors(word, outputWordLen, alphabetLetters))

        if len(inputs) != 0:
            yield inputs, outputs
