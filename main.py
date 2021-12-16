import numpy as np
import models
testModel = models.loadModel('hands') # models.trainNewModel(10, 'hands') 
loss, accuracy = models.getLossAndAccuracy(testModel)
print(f'Model loss: {round(loss*100, 2)}%, Model Accuracy: {round(accuracy*100, 2)}%')
for i in range(10):
    print("==========================================")
    imagePath = f'prepared/digit{i}.png'
    prediction = models.getPrediction(testModel, imagePath)

    print(f"Prediction of {imagePath}:")
    bestGuess = np.argmax(prediction)
    print(f'Digit is most likely: {bestGuess}')

# imagePath_err = 'prepared/digitE.png'
# prediction_err = models.getPrediction(testModel, imagePath_err)
# print(f"Prediction of {imagePath_err}:")
# bestGuess_err = np.argmax(prediction_err)
# print(f'Digit is most likely: {bestGuess_err}')

# print("==========================================")

# imagePath_non = 'prepared/digitN.png'
# prediction_non = models.getPrediction(testModel, imagePath_non)
# print(f"Prediction of {imagePath_non}:")
# bestGuess_non = np.argmax(prediction_non)
# print(f'Digit is most likely: {bestGuess_non}')

