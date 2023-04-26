import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.abspath('../yolov5'))
from utils.general import non_max_suppression, scale_coords
# from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
from typing import List
# from dynaconf import settings
from models.experimental import attempt_load
import cv2

class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        cwd = os.path.dirname(__file__)
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=size
        
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

    def detect(self, frame):
        
        results, resized_img = self.char_detection_yolo(frame)

        return results, resized_img
    
    def preprocess_image(self, original_image):

        resized_img = self.ResizeImg(original_image,size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, classes=None, \
                            agnostic_nms=True, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            # det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    # xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf), (xyxy[0],xyxy[1],xyxy[2],xyxy[3])]
                    results.append(result)
        # print(results)
        return results, resized_img
        
    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        # print(h1, w1, _)
        h, w = size
        if w1 < h1 * (w / h):
            # print(w1/h1)
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
    def load_model(self,path, train = False):
        # print(self.device)
        model = attempt_load(path, map_location=self.device)  # load FP32 model
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
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='Vietnamese_imgs', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt




if __name__ == '__main__':
    opt = parse_opt()
    
    char_model=Detection(size=opt.imgsz,weights_path=opt.weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    path=opt.source

    img_names=os.listdir(path)

    for img_name in img_names:
        img=cv2.imread(os.path.join(path,img_name))
        results, resized_img=char_model.detect(img.copy())
        for name,conf,box in results:
            resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 2)
            resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
        if not os.path.exists(os.path.join('out')):
            os.makedirs(os.path.join('out'))
        cv2.imwrite(os.path.join('out',img_name),resized_img)
