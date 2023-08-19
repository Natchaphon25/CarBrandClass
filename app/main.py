import pickle
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  
    allow_methods=['*'],  
    allow_headers=['*']   
)

@app.post("/api/brandofcar")
async def read_image(image: Request):

    url_genHog = 'http://172.17.0.2:80/api/genhog/'
    
    file_model_path = r"/model/ClassifierCarModel.pkl"
   
    with open(os.getcwd()+file_model_path, "rb") as file:
        carPredictor = pickle.load(file) # อ่าน model ที่ได้ทำการเรียนรู้ไว้
    
    # carPredictor = pickle.load(open(os.getcwd()+'ClassifierCarModel.pkl', 'rb')) 

    data = await image.json() # สร้าง json
    # print(data)

    # เรียกใช้ api โดยส่ง base64 ที่รับมานั้น ไปอีกทีเพื่อหาเอกลักษณ์ของรูปภาพค่า HOG
    hog = requests.post(url_genHog, json=data)
    # print(hog["HOG Descriptor"])
    # if hog:
        # print(hog.json()['HOG Descriptor'])
    # # ค่าที่ตอบ hog กลับมานั้นจะเป็นแบบ json ดังนั้นต้องแปลงให้เป็น list
    # # ค่าที่ตอบกลับมา จะมี 2 ค่า คือ HOG Length และ HOG vector แต่เราต้องการแค่ HOG vector
    hog = hog.json()['HOG Descriptor']
    result = carPredictor.predict([hog])
    print(result[0])
    return {'Brand of this car is': result[0]}
    # if result:

    #     return {"dd": result[0]}
    # else:
    #     return {"ss"}