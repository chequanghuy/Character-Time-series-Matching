import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath('yolov5'))
from yolov5.utils.general import non_max_suppression, scale_coords
from typing import List
from yolov5.models.experimental import attempt_load
import cv2
import math
class CharacterDetection:
    def __init__(self, weights_path=None, model_name='character_detection'):
        cwd = os.path.dirname(__file__)
        
        label_path = './character_name.txt'

        self.char_model, names_ = self.load_model('exp/weights/best.pt')
        self.names = [name.strip() for name in open(label_path).readlines()]
        self.size=128

    def detect(self, frame,agnostic_nms=False):
        
        results, resized_img = self.char_detection_yolo(frame,agnostic_nms=agnostic_nms)

        return results, resized_img
    
    def preprocess_image(self, original_image, size=(128, 128), device='cuda'):

        resized_img = self.ResizeLetter(original_image,size)
        # print("resized_img.shape",resized_img.shape)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # print("resized_img.shape",resized_img.shape)
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, conf_thres=0.05, \
                            iou_thres=0.01, classes=None, \
                            agnostic_nms=False, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=conf_thres,
                                            iou_thres=iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf), (xc,yc,w_,h_)]
                    results.append(result)
        # print(results)
        return results, resized_img

    def ResizeLetter(self,img, size,stride=64):
        h1, w1, _=img.shape
        h,w= size
        if w1 < h1*(w/h):
            char_digit = cv2.resize(img, (int(float(w1/h1)*h), h),cv2.INTER_LANCZOS4)
            a=math.ceil(int(float(w1/h1)*h)/stride)
            b=a*stride-int(float(w1/h1)*h)
            mask1 = np.full((h, b//2, 3),114, np.uint8)
            mask2 = np.full((h, b-b//2, 3),114, np.uint8)
            thresh = cv2.hconcat([mask2, char_digit, mask1])
            return thresh
        else:
            char_digit = cv2.resize(img, (w, int(float(h1/w1)*w)),cv2.INTER_LANCZOS4)
            a=math.ceil(int(float(h1/w1)*w)/stride)
            b=a*stride-int(float(h1/w1)*w)
            mask1 = np.full((b//2, w, 3),114, np.uint8)
            mask2 = np.full((b-b//2, w, 3),114, np.uint8)
            thresh = cv2.vconcat([mask2, char_digit, mask1])
            return thresh    

    def load_model(self,path, train = False):
        model = attempt_load(path, map_location='cuda')  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if train:
            model.train()
        else:
            model.eval()
        return model, names
    def xyxytoxywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y
    
    
char = CharacterDetection()

