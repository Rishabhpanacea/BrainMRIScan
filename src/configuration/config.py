JPEG_TEMP_PATH = "JPEG_TEMP_PATH"
DICOM_TEMP_PATH = "DICOM_TEMP_PATH"

NUM_CLASSES: int = 2  # including background.
CLASSES: list = ["Bleed"]
legac_ohif_text = {"Bleed": "ICH-A"}
IMAGE_SIZE: tuple[int, int] = (512, 512)  # W, H
MEAN: tuple = (0.485, 0.456, 0.406)
STD: tuple = (0.229, 0.224, 0.225)
PIXEL_THRESHOLD = 0