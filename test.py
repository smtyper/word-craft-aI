import tensorflow as tf
import numpy as np
import dataset

from model import alphabetLetters, modelName, inputWordLen

model = tf.keras.models.load_model(modelName)

testInputs = {"авт": "", "агр": "", "адм": "", "азб": "", "ази": "", "акт": "", "але": "", "алм": "", "алф": "", "амб": "", "амф": "", "анг": "", "ант": "", "апт": "", "арб": "", "арк": "", "арм": "", "арт": "", "асп": "", "аст": "", "ато": "", "ауд": "", "аук": "", "бак": "", "бал": "", "бан": "", "бар": "", "бас": "", "бат": "", "бег": "", "бед": "", "без": "", "бел": "", "бер": "", "бес": "", "бил": "", "бир": "", "бис": "", "бит": "", "бле": "", "бли": "", "бло": "", "блю": "", "бог": "", "бод": "", "бок": "", "бол": "", "бом": "", "бор": "", "бот": "", "боя": "", "бре": "", "бри": "", "бро": "", "бру": "", "буд": "", "бук": "", "бур": "", "бут": "", "ваг": "", "вал": "", "ван": "", "вар": "", "вас": "", "ват": "", "вах": "", "век": "", "вел": "", "вен": "", "вер": "", "вес": "", "вет": "", "вид": "", "виз": "", "вил": "", "вин": "", "вис": "", "вит": "", "вих": "", "вкл": "", "вла": "", "вли": "", "вло": "", "вну": "", "вод": "", "вой": "", "вок": "", "вол": "", "воп": "", "вос": "", "вот": "", "выб": "", "выг": "", "выд": "", "вым": "", "вып": "", "выс": "", "выт": "", "выш": "", "гад": "",}
# testInputs = {"чел": ""}


for testInput in testInputs.keys():
    testInputVector = dataset.toVectors(testInput, inputWordLen, alphabetLetters)
    testInputBatch = np.stack([testInputVector])

    predictions = model.predict(testInputBatch)[0]
    results = ["".join([alphabetLetters[np.argmax(charVector)] for charVector in prediction])
              for prediction in predictions]

    testInputs[testInput] = [(testInput + result).replace(" ", "")for result in results]

    print()

resultStr = "\n".join([f"{key}: {value}" for key, value in testInputs.items()])