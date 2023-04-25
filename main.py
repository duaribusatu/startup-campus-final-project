import torch
# import numpy as np
# import torchvision.transforms as T
from PIL import Image
import cv2 as cv
# import json
# import requests
# import io
import base64

#file from other module
from init import init_model
from preprocessing import grayscaling
from inference import get_result, upnoscale
from postprocessing import get_postprocessing_name

#############################################################################################

def ai_pipeline(model_inf, img_path, image_path):
    img = grayscaling(img_path)

    image_tensor  = upnoscale(img_path = img, image_path=image_path)

    image_tensor.resize_(1, 3, 224, 224)

    output_numpy = get_result(image= image_tensor, model_inf=model_inf)

    highest = get_postprocessing_name(output_model=output_numpy)

    return highest

def main():
    # img_data = b'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'

    # with open("JuryTest/sandal1.png", "wb") as fh:
    #         fh.write(base64.decodebytes(img_data))

    #model initialization

    # image_base64 = requests.data["image64"]

    # converting from base64 to image cv
    # base64 processing to cv image
    # reconstruct image as an numpy array
    # image64_decode = cv.imread(io.BytesIO(base64.b64decode(image_base64)))

    # # finally convert RGB image to BGR for opencv
    # # and save result
    # image_cv = cv.cvtColor(image64_decode, cv.COLOR_RGB2BGR) 

    device = torch.device('cpu')
    #initial model
    path_model = "weight/resnet18_custom_11_class_fix4.pth" #changed everytime maybe
    inf_model = init_model(model_path = path_model , device = device)

    #preprocessing 
    image_path = "JuryTest/mnist_358.png"
    img_cv = cv.imread(image_path)

    #requests from API
    highest = ai_pipeline(model_inf = inf_model, img_path = img_cv, 
                        image_path = image_path)

    print("highest score label : ", highest)

    # # for API
    # response = { 
    #     "classification_name": result_name,
    #     "classification_score": result_score
    # }

if __name__  == "__main__":
    print("Classification Program")
    main()

