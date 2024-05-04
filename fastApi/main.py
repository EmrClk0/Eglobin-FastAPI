from typing import Union
from fastapi import FastAPI, File,UploadFile, HTTPException, status
from fastapi.responses import FileResponse
import uuid
from ultralytics import YOLO
import cv2

from YOLOV8predict import extraxtROI


app = FastAPI()

IMAGE_DIR = "images/"
CROPPED_DIR = "cropped/"
accepted_file_types = ["image/png", "image/jpeg", "image/jpg", "png", "jpeg", "jpg"] 

weightPath="ptweights/best200.pt"
confidence=0.8


@app.get("/")
def deeneme():
    return {"hi":"world"}


#segment edebilrse segmen ettiği resmi döndür
#segment edemesse yok de

#segment edilirse kullanıcı onay verecek haliyle segment pathini döndür ve sonraki croping işlemi için 

#göz tespit algoritması eklenmeli 
#APİ KEY !

@app.post("/upload/")
async def uploadImage(file: UploadFile=File(...)):
    content_type = file.content_type

    if content_type in accepted_file_types:
        file.filename=f"{uuid.uuid4()}{file.filename}"
        contents = await file.read()
    
        #saving file
        with open(f"{IMAGE_DIR}{file.filename}","wb") as f:
            f.write(contents)
        
        model = YOLO(weightPath)
        imageSource=IMAGE_DIR+file.filename
        image = cv2.imread(imageSource)
        result = extraxtROI(model,confidence,image,CROPPED_DIR)
        
        if(result != None):
            segment, cropped = result
            return {"segmentedImagePATH":segment,
                    "croppedImagePATH":cropped}
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="ROI DOES NOT EXİST")

        return {"filename":file.filename}
        #end of if
    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"{content_type} is not supported format")
    
    
@app.get("/getImage")  #get ımage
def getImageByPath(imagePath:str):
    return FileResponse(imagePath)