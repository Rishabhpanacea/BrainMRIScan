import os.path
import tempfile
from fastapi import APIRouter, UploadFile
from src.models.transformers import TUMOR_MODEL
import os
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import FileResponse
# from src.models.transformers import predict
from fastapi.responses import JSONResponse
import numpy as np
from src.configuration.config import DICOM_TEMP_PATH
from src.utils import ResponseUtils
from src.utils.DicomUtils import saveDicomAsJPEG, getDicomFIleIds
from glob import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

router = APIRouter()

@router.post("/predict/")
async def create_prediction(file: UploadFile):
    print("I am in the function")
    model = TUMOR_MODEL()
    fd, dicom_path = tempfile.mkstemp(suffix=".dcm", dir=DICOM_TEMP_PATH, prefix='tmp')
    with os.fdopen(fd, 'wb') as tmp:
        data = await file.read()
        tmp.write(data)
    jpeg_path = saveDicomAsJPEG(dicom_path)
    sop_instance_uid, study_instance_uid, series_instance_uid \
        = getDicomFIleIds(dicom_path)
    seg_info = model.predict(jpeg_path) 
    response = ResponseUtils.getOHIFObjects(sop_instance_uid, study_instance_uid, series_instance_uid, seg_info)

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    serializable_response = make_serializable(response)

    return JSONResponse(content=serializable_response)


# @router.post("/upload-tif/")
# async def upload_tif(file: UploadFile = File(...)):
#     if file.content_type not in ["image/tiff", "image/tif"]:
#         return JSONResponse(content={"message": "Invalid file type. Please upload a .tif file."}, status_code=400)
#     try:
#         # Read the file contents into a numpy array
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

#         if image is None:
#             return JSONResponse(content={"message": "Could not process the image."}, status_code=400)

#         # Perform any processing on the image if needed
#         height, width = image.shape[:2]

#         return {"filename": file.filename, "width": width, "height": height, "dtype": str(image.dtype)}
#     except Exception as e:
#         return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)






# @router.post("/predict/")
# async def create_prediction(file: UploadFile):
#     print("I am in the function")
#     file_location = f"./uploads/{file.filename}"
#     with open(file_location, "wb+") as file_object:
#         file_object.write(file.file.read())
#     print(file)    
#     return JSONResponse(content={"filename": file.filename})
    # data = await file.read()
    # print(data)
    # model = TUMOR_MODEL()
    # fd, tif_path = tempfile.mkstemp(suffix=".tif", dir=TIFF_TEMP_PATH, prefix='tmp')
    # with os.fdopen(fd, 'wb') as tmp:
    #     data = await file.read()
    #     print(data)
    #     tmp.write(data)
    # jpeg_path = saveImageAsJPEG(tif_path)
    # sop_instance_uid, study_instance_uid, series_instance_uid \
    #     = getImageFileIds(tif_path)
    # seg_info = model.predict(jpeg_path) 
    # response = ResponseUtils.getOHIFObjects(sop_instance_uid, study_instance_uid, series_instance_uid, seg_info)

    # def make_serializable(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     if isinstance(obj, dict):
    #         return {k: make_serializable(v) for k, v in obj.items()}
    #     if isinstance(obj, list):
    #         return [make_serializable(i) for i in obj]
    #     return obj

    # serializable_response = make_serializable(response)

    # return JSONResponse(content=serializable_response)

# @router.post("/predict/")
# async def create_prediction(file: UploadFile):
#     print("I am in the function")
#     model = TUMOR_MODEL()
#     fd, dicom_path = tempfile.mkstemp(suffix=".dcm", dir=DICOM_TEMP_PATH, prefix='tmp')
#     with os.fdopen(fd, 'wb') as tmp:
#         data = await file.read()
#         print(data)
#         tmp.write(data)
#     jpeg_path = saveDicomAsJPEG(dicom_path)
#     sop_instance_uid, study_instance_uid, series_instance_uid \
#         = getDicomFIleIds(dicom_path)
#     seg_info = model.predict(jpeg_path) 
#     response = ResponseUtils.getOHIFObjects(sop_instance_uid, study_instance_uid, series_instance_uid, seg_info)

#     def make_serializable(obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, dict):
#             return {k: make_serializable(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [make_serializable(i) for i in obj]
#         return obj

#     serializable_response = make_serializable(response)

#     return JSONResponse(content=serializable_response)





# @router.post("/predict/")
# async def create_prediction(request: Request):
#     print("INside post request")
#     im_width = 256
#     im_height = 256
    

#     data = await request.json()
#     image_path = data.get("image_path")
#     print("image path",image_path)
    
#     if not image_path:
#         return JSONResponse(content={"error": "Image path not provided"}, status_code=400)

#     print(f"Received image path: {image_path}")
#     model = TUMOR_MODEL()

#     img = cv2.imread(image_path)
#     img = cv2.resize(img ,(im_height, im_width))
#     img = img / 255
#     img = img[np.newaxis, :, :, :]
#     pred=model.predict_seg(image_path)

#     plt.figure(figsize=(12,12))
#     plt.subplot(1,3,1)
#     plt.imshow(np.squeeze(img))
#     plt.title('Original Image')
#     # plt.subplot(1,3,2)
#     # plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
#     # plt.title('Original Mask')
#     plt.subplot(1,3,2)
#     plt.imshow(np.squeeze(pred) > .5)
#     plt.title('Prediction')
#     plt.show()

#     return "hellow"




# @router.post("/predict/")
# async def create_prediction(request: Request):
#     im_width = 256
#     im_height = 256


#     data = await request.json()
#     image_path = data.get("image_path")
    
#     if not image_path:
#         return JSONResponse(content={"error": "Image path not provided"}, status_code=400)

#     print(f"Received image path: {image_path}")
#     model = TUMOR_MODEL()

#     img = cv2.imread(image_path)
#     img = cv2.resize(img ,(im_height, im_width))
#     img = img / 255
#     img = img[np.newaxis, :, :, :]
#     pred=model.predict_seg(image_path)

#     plt.figure(figsize=(12,12))
#     plt.subplot(1,3,1)
#     plt.imshow(np.squeeze(img))
#     plt.title('Original Image')
#     # plt.subplot(1,3,2)
#     # plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
#     # plt.title('Original Mask')
#     plt.subplot(1,3,2)
#     plt.imshow(np.squeeze(pred) > .5)
#     plt.title('Prediction')
#     plt.show()

#     return "hellow"










































    
# @router.post("/predict/")
# async def create_prediction(file: UploadFile):
#     im_width = 256
#     im_height = 256
#     train_files = []
#     mask_files = glob("C:/Users/Rishabh/Downloads/archive/lgg-mri-segmentation/kaggle_3m/*/*_mask*")

#     for i in mask_files:
#         train_files.append(i.replace('_mask',''))


#     df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
#     df_train, df_test = train_test_split(df,test_size = 0.1)
#     df_train, df_val = train_test_split(df_train,test_size = 0.2)
#     print(df_train.values.shape)
#     print(df_val.values.shape)
#     print(df_test.values.shape)    
#     model = TUMOR_MODEL()



#     for i in range(30):
#         index=np.random.randint(1,len(df_test.index))
#         img = cv2.imread(df_test['filename'].iloc[index])
#         img = cv2.resize(img ,(im_height, im_width))
#         img = img / 255
#         img = img[np.newaxis, :, :, :]
#         pred=model.predict(img)
#         plt.figure(figsize=(12,12))
#         plt.subplot(1,3,1)
#         plt.imshow(np.squeeze(img))
#         plt.title('Original Image')
#         plt.subplot(1,3,2)
#         plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
#         plt.title('Original Mask')
#         plt.subplot(1,3,3)
#         plt.imshow(np.squeeze(pred) > .5)
#         plt.title('Prediction')
#         plt.show()    

#     return "hellow"   
