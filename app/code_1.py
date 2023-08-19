import json
import pickle


file_path = "hogTest.json"
with open(file_path, "rb") as file:
    hog = json.load(file)

print(hog["HOG Descriptor"])

file_model_path = "model\ClassifierCarModel.pkl"
with open(file_model_path, "rb") as file:
    modelRead = pickle.load(file)

result = modelRead.predict([hog["HOG Descriptor"]])
print(result[0])