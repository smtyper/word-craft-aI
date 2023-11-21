import dataset
import numpy as np

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Reshape

alphabetLetters = " абвгдеёжзийклмнопстуфхцчшщъыьэюя"
inputWordLen = 3

outputWordCount = 2
outputWordLen = 15
modelName = "v0.3"

if __name__ == "__main__":
    model = Sequential()
    model.add(Input(shape = (inputWordLen, len(alphabetLetters))))

    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(512, return_sequences = True))
    model.add(LSTM(512))

    model.add(Dense(512, activation = "relu"))
    model.add(Dense(outputWordCount * outputWordLen * len(alphabetLetters), activation = "softmax"))

    model.add(Reshape((outputWordCount, outputWordLen, len(alphabetLetters))))

    model.compile(optimizer = "adam", 
                loss = "categorical_crossentropy",
                metrics = ["accuracy"])
    model.summary()

    batch_size = 64
    epochs = 50

    nouns = dataset.getNounsWords(minLen = 3, maxLen = 6)
    splittedWords = dataset.splitWords(nouns, wordsInGroupCount = outputWordCount)
    trainingSet = dataset.fitDataset(splittedWords, alphabetLetters, inputWordLen, outputWordCount,
                                    outputWordLen,
                                    batchSize = 64,
                                    epochs = epochs)


    history = model.fit(trainingSet,
                        steps_per_epoch = len(splittedWords) // batch_size,
                        epochs = epochs)
                        # validation_data = testSet,
                        # validation_steps = len(testIndices) // batch_size)
    model.save(modelName, overwrite = True)