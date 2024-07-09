from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from src.utils.Architecture import dice_coef,dice_coef_loss,iou,jac_distance

class TUMOR_MODEL():
    def __init__(self):
        self.model_path = r'C:\Users\Rishabh\Downloads\MRIOFBRAIN.h5'
        self.model = load_model(self.model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    def predict_seg(self,image_path):
        im_width = 256
        im_height = 256
        img = cv2.imread(image_path)
        print((self.model.input_shape[1], self.model.input_shape[2]))
        img = cv2.resize(img, (self.model.input_shape[1], self.model.input_shape[2]))  # Resize the image to the input shape expected by the model
        img_array = np.expand_dims(img, axis=0)  
        img = cv2.imread(image_path)
        img = cv2.resize(img ,(im_height, im_width))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        prediction = self.model.predict(image_path)
        # plt.figure(figsize=(12,12))
        # plt.subplot(1,3,1)
        # plt.imshow(np.squeeze(img))
        # plt.title('Original Image')
        # plt.subplot(1,3,2)
        # plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
        # plt.title('Original Mask')
        # plt.subplot(1,3,2)
        # plt.imshow(np.squeeze(prediction) > .5)
        # plt.title('Prediction')
        # plt.show()

    #      index=np.random.randint(1,len(df_test.index))
    # img = cv2.imread(df_test['filename'].iloc[index])
    # img = cv2.resize(img ,(im_height, im_width))
    # img = img / 255
    # img = img[np.newaxis, :, :, :]
    # pred=model.predict(img)

        return prediction    














# smooth = 100

# def dice_coef(y_true, y_pred):
#     y_truef = K.flatten(y_true)
#     y_predf = K.flatten(y_pred)
#     And = K.sum(y_truef * y_predf)
#     return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# def iou(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred)
#     sum_ = K.sum(y_true + y_pred)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return jac

# def jac_distance(y_true, y_pred):
#     y_truef = K.flatten(y_true)
#     y_predf = K.flatten(y_pred)
#     return -iou(y_true, y_pred)

# # Update the path to your model file correctly
# # model_path = r'C:\Users\risha\Downloads\MRIOFBRAIN.h5'

# model_path = r'C:\Users\Rishabh\Downloads\MRIOFBRAIN.h5'

# model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

# # Add a prediction function
# def predict(image_path):
#     # Load and preprocess the image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (model.input_shape[1], model.input_shape[2]))  # Resize the image to the input shape expected by the model
#     img_array = np.expand_dims(img, axis=0)  # Add batch dimension

#     # Perform prediction
#     prediction = model.predict(img_array)
#     # print(prediction)

#     return prediction

    # Postprocess the prediction (e.g., convert to binary mask)
    # prediction = (prediction > 0.5).astype(np.uint8)

    # Visualize the input image and the prediction
    # plt.subplot(1, 2, 1)
    # plt.title("Input Image")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # plt.subplot(1, 2, 2)
    # plt.title("Prediction")
    # plt.imshow(prediction[0], cmap='gray', alpha=0.5)

    # plt.show()

# Example usage
# image_path = r'path_to_your_image.jpg'  # Update with the path to your image
# predict(image_path)

# def classify(self, jpeg_path):
#     image = Image.open(jpeg_path).convert('RGB')
#     shape_H_W = image.size[::-1]
#     input_tensor = self._val_transforms(image)
#     input_tensor = input_tensor.unsqueeze(0).to(self.device)

#     with torch.no_grad():
#         outputs = self.model(pixel_values=input_tensor.to(self.device), return_dict=True)
#         predictions = F.interpolate(outputs["logits"], size=shape_H_W, mode="bilinear", align_corners=False)
#         preds_argmax = predictions.argmax(dim=1).cpu().squeeze().numpy()
#         seg_info = [{"seg":np.array(preds_argmax == idx, dtype=np.int64).tolist(), "seg_class": class_name, "pathology_present":str((preds_argmax == idx).sum() > PIXEL_THRESHOLD)} for idx, class_name in enumerate(CLASSES, 1)]
#         return seg_info
    












# Making Segmentation onject using images dir and model output masks

# img = Image.open("/content/sample_data/mask.jpg")
# mask_arr = np.array(img)

# from pathlib import Path

# import highdicom as hd
# import numpy as np
# from pydicom.sr.codedict import codes
# from pydicom.filereader import dcmread
# from uuid import uuid4


# # Path to directory containing single-frame legacy CT Image instances
# # stored as PS3.10 files
# series_dir = Path('/content/101453181 SEEMA/RADIWMC001297215 Head AIIMS_DF_HeadRoutineSeq Adult/CT HeadSeq 1.0 H30s')
# image_files = series_dir.glob('*.dcm')

# # Read CT Image data sets from PS3.10 files on disk
# image_datasets = [dcmread(str(f)) for f in image_files]
# description_segment =[]
# masks = []

# for index,dcm_im in enumerate(image_datasets):
#   uuid = str(uuid4())
#   uid = hd.UID.from_uuid(uuid)
#   mask = np.array(mask_arr, dtype=bool)
#   masks.append(mask)


#   # Describe the algorithm that created the segmentation
#   algorithm_identification = hd.AlgorithmIdentificationSequence(
#       name='test',
#       version='v1.0',
#       family=codes.cid7162.ArtificialIntelligence
#   )

#   # Describe the segment
#   description_segment_1 = hd.seg.SegmentDescription(
#       segment_number=1,
#       segment_label='segment {}'.format(index +1),
#       segmented_property_category=codes.cid7150.Tissue,
#       segmented_property_type=codes.cid7166.ConnectiveTissue,
#       algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
#       algorithm_identification=algorithm_identification,
#       tracking_uid=uid,
#       tracking_id='test segmentation of computed tomography image {}'.format(index + 1)
#   )
#   description_segment.append(description_segment_1)

# uuid = str(uuid4())
# uid = hd.UID.from_uuid(uuid)
#   # Create the Segmentation instance
# seg_dataset = hd.seg.Segmentation(
#     source_images=image_datasets,
#     pixel_array=np.array(masks),
#     segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
#     segment_descriptions=[description_segment_1],
#     series_instance_uid=uid,
#     series_number=2,
#     sop_instance_uid=uid,
#     instance_number=index +1,
#     manufacturer='Manufacturer',
#     manufacturer_model_name='Model',
#     software_versions='v1',
#     device_serial_number='Device XYZ',
# )

# # print(seg_dataset)

# seg_dataset.save_as("/content/sample_data/segmask_multiple.dcm".format(index))
