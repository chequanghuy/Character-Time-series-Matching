import argparse
import os
import glob
import random

import cv2
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import process_plate
from Char_detection_yolo import CharacterDetection
import imutils


def main():
    path='./test_tracks/'
    plate_folders = sorted(os.listdir(path))
    char_model=CharacterDetection()
    
    count=0
    acc=0
    case_fail=[]
    with open("./test_results.txt","r") as f:
        labels=f.read().split("\n")
    from tqdm import tqdm
    for count,fd in enumerate(tqdm(plate_folders)):
        alpha=0
        results=""
        track_boxs=[]
        Ws=[]
        Hs=[]
        
        plate_images=[f for f in os.listdir(os.path.join(path,fd)) if f.endswith("png")]
        plate_images = sorted(plate_images)
        if len(plate_images)==0:
            continue
        
        for index_,f in enumerate(plate_images):
            
            image_name=os.path.join(path,fd,f)
            image = cv2.imread(image_name)
            h,w,_=image.shape
            Ws.append(w)
            Hs.append(h)
            image = imutils.rotate(image, math.degrees(alpha))
            detections,image=char_model.detect(image)
            if len(detections)==0:
                continue
            
            dets=process_plate.merge_box(detections)
            track_box=[]

            for label, confidence, box in dets:
                track_box.append([(int(round(box[0] - box[2]/2))), (int(round(box[1] - box[3]/2))), (int(round(box[0] + box[2]/2))), (box[1] + box[3]/2),
                                       [[float(c)] for c in confidence.split("-")],[[l] for l in label.split("-")]])
            track_box=np.array(track_box,dtype=object)
            center_x = (track_box[:, 0] + track_box[:, 2]) / 2
            center_y = (track_box[:, 1] + track_box[:, 3]) / 2
            track_boxs.append(track_box)
            
            center = np.vstack((center_x, center_y)).T

            degree=process_plate.find_angle(center_x,center_y)

            if 3<abs(math.degrees(degree))<25:
                alpha-=degree

        old_char = np.zeros((0, 0))
        for track_box in track_boxs:
            arr_track=process_plate.matching_char(old_char,track_box)
            old_char=arr_track
        Hm=np.mean(np.array(Hs)) if len(Hs)>0 else 0
        Wm=np.mean(np.array(Ws)) if len(Ws)>0 else 0

        if arr_track.shape[0]>7:
            arr_track=process_plate.merge_box_arr_track(arr_track)
        arr_track=sorted(arr_track, key=lambda x: float(x[0]))
        re=""
        arr_track=np.array([arr_ for arr_ in arr_track if len(arr_[5])>=1/2*(len(plate_images))],dtype=object)
        for arr_ in arr_track:

            clss=max(arr_[5],key=arr_[5].count)
            clss=process_plate.get_maximum_conf_char(arr_)   
            re+=clss         
        if Hm*2>Wm:
            center_x = (arr_track[:, 0] + arr_track[:, 2]) / 2
            center_y = (arr_track[:, 1] + arr_track[:, 3]) / 2
            chars = ["{}".format(process_plate.get_maximum_conf_char(track_box_)) for track_box_ in arr_track]
            _,re=process_plate.find_chars_plate(center_x,center_y,chars)
        re=re.replace("-","")
        label=labels[count].replace("-","")
        re=re[0:3].replace("0","O").replace("1","I")+re[3:]
        # print(fd,re)
        if re == label:
            acc+=1
        else:
            case_fail.append(fd)
        # count+=1

    print(acc/60)
    print("case fail",case_fail)
        

                


if __name__ == "__main__":
    main()
