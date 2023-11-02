import dataset
import numpy as np

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Reshape

alphabetLetters = " абвгдеёжзийклмнопстуфхцчшщъыьэюя"
inputWordLen = 3
outputWordLen = 15

model = Sequential()
model.add(Input(shape = (inputWordLen, len(alphabetLetters))))

model.add(LSTM(256, return_sequences = True))
model.add(LSTM(512, return_sequences = True))
model.add(LSTM(512))

model.add(Dense(512, activation = "relu"))
model.add(Dense(outputWordLen * len(alphabetLetters), activation = "softmax"))

model.add(Reshape((outputWordLen, len(alphabetLetters))))

model.compile(optimizer = "adam", 
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])
model.summary()

batch_size = 64
epochs = 10

nouns = dataset.getNounsWords(minLen = 5, maxLen = 10)
trainingIndices, testIndices = dataset.getSetsIndices(len(nouns))
trainingSet = dataset.fitDataset(nouns, trainingIndices, alphabetLetters, inputWordLen, outputWordLen, batchSize = 64,
                                 epochs = epochs)
testSet = dataset.fitDataset(nouns, testIndices, alphabetLetters, inputWordLen, outputWordLen, batchSize = 64,
                             epochs = epochs)

history = model.fit(trainingSet,
                    steps_per_epoch = len(trainingIndices) // batch_size,
                    epochs = epochs,
                    validation_data = testSet,
                    validation_steps = len(testIndices) // batch_size)

testInputs = {"авт": "", "агр": "", "адм": "", "азб": "", "ази": "", "акт": "", "але": "", "алм": "", "алф": "", "амб": "", "амф": "", "анг": "", "ант": "", "апт": "", "арб": "", "арк": "", "арм": "", "арт": "", "асп": "", "аст": "", "ато": "", "ауд": "", "аук": "", "бак": "", "бал": "", "бан": "", "бар": "", "бас": "", "бат": "", "бег": "", "бед": "", "без": "", "бел": "", "бер": "", "бес": "", "бил": "", "бир": "", "бис": "", "бит": "", "бле": "", "бли": "", "бло": "", "блю": "", "бог": "", "бод": "", "бок": "", "бол": "", "бом": "", "бор": "", "бот": "", "боя": "", "бре": "", "бри": "", "бро": "", "бру": "", "буд": "", "бук": "", "бур": "", "бут": "", "ваг": "", "вал": "", "ван": "", "вар": "", "вас": "", "ват": "", "вах": "", "век": "", "вел": "", "вен": "", "вер": "", "вес": "", "вет": "", "вид": "", "виз": "", "вил": "", "вин": "", "вис": "", "вит": "", "вих": "", "вкл": "", "вла": "", "вли": "", "вло": "", "вну": "", "вод": "", "вой": "", "вок": "", "вол": "", "воп": "", "вос": "", "вот": "", "выб": "", "выг": "", "выд": "", "вым": "", "вып": "", "выс": "", "выт": "", "выш": "", "гад": "",}

for testInput in testInputs.keys():
    testInputVector = dataset.toVectors(testInput, 3, alphabetLetters)
    testInputBatch = np.stack([testInputVector])

    prediction = model.predict(testInputBatch)[0]
    result = "".join([alphabetLetters[np.argmax(charVector)] for charVector in prediction])

    testInputs[testInput] = (testInput + result[3:]).replace(" ", "")

    print()

resultStr = "\n".join([f"{key}: {value}" for key, value in testInputs.items()])

print()