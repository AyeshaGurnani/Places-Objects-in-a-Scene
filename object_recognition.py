from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.4}

tfnet = TFNet(options)

def run(im_name):

    print im_name
    imgcv = cv2.imread(im_name)
    print imgcv.shape
    result = tfnet.return_predict(imgcv)
    # result_ = []
    # for obj in result:
    #     if obj['confidence'] > 0.4:
    return result
    # print(result)