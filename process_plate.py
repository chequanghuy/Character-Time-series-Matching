from cmath import atan
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import cv2
import torch
def get_maximum_conf_char(track_box):
    
    conf_list, cls_list = np.array(track_box[4]), np.array(track_box[5])
    unique_cls = np.unique(cls_list)

    sum_conf = np.array([conf_list[np.where(cls_list == i)[0]].sum() for i in unique_cls])
    
    maximum_conf_index = np.argmax(sum_conf)
    maximum_conf_cls = unique_cls[maximum_conf_index]
    return maximum_conf_cls
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
def box_iou(box1, box2):

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def estimate_coef(x, y):
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)
  
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    a = SS_xy / SS_xx
    b = m_y - a*m_x
  
    return (a,b)
def find_angle(center_x,center_y):
    a,b = estimate_coef(center_x,center_y)
    return -math.atan(a)
def matching_char(character_storage, new_characters):
    def distance(a, b):
        dis = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return dis

    def linear_assignment(cost_matrix, maximize=False):
        x, y = linear_sum_assignment(cost_matrix, maximize)
        return np.array(list(zip(x, y)))


    if character_storage.shape[0] == 0:
        return new_characters


    if new_characters.shape[0] == 0:
        return character_storage

    tensor_character_storage = np.array(character_storage[:, :4], dtype=np.float32)
    tensor_new_characters = np.array(new_characters[:, :4], dtype=np.float32)

    center_x_1 = (tensor_character_storage[:, 0] + tensor_character_storage[:, 2]) / 2
    center_y_1 = (tensor_character_storage[:, 1] + tensor_character_storage[:, 3]) / 2
    center_character_storage = np.vstack((center_x_1, center_y_1)).T
    
    center_x_2 = (tensor_new_characters[:, 0] + tensor_new_characters[:, 2]) / 2
    center_y_2 = (tensor_new_characters[:, 1] + tensor_new_characters[:, 3]) / 2
    center_new_characters = np.vstack((center_x_2, center_y_2)).T
    

    
    dis_mat = np.zeros((len(center_character_storage), len(center_new_characters)), dtype=np.float32)
    for t, trk in enumerate(center_character_storage):
        for d, det in enumerate(center_new_characters):

            dis_mat[t, d] = distance(trk, det)

    if dis_mat.size != 0:
        matched_idx = linear_assignment(dis_mat)
    else:
        matched_idx = np.empty((0, 2))

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(character_storage):
        if t not in matched_idx[:, 0]:
            unmatched_trackers.append(t)

    for d, det in enumerate(new_characters):
        if d not in matched_idx[:, 1]:
            unmatched_detections.append(d)

    matches = []
    for m in matched_idx:
        if dis_mat[m[0], m[1]] > 35:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    delta_v = [0, 0]
    for m in matches:
        delta_v[0] += new_characters[m[1]][0] + new_characters[m[1]][2] \
                      - character_storage[m[0]][0] - character_storage[m[0]][2]
        delta_v[1] += new_characters[m[1]][1] + new_characters[m[1]][3] \
                      - character_storage[m[0]][1] - character_storage[m[0]][3]

        character_storage[m[0]][0:4] = new_characters[m[1]][0:4]
        character_storage[m[0]][4] += new_characters[m[1]][4]
        character_storage[m[0]][5] += new_characters[m[1]][5]

    if len(matches) > 0:
        delta_v[0] = delta_v[0] / len(matches)
        delta_v[1] = delta_v[1] / len(matches)

    if len(unmatched_detections) != 0:
        character_storage = np.append(character_storage, new_characters[unmatched_detections], axis=0)
    for u in unmatched_trackers:
        character_storage[u][0] = character_storage[u][0] + delta_v[0]
        character_storage[u][2] = character_storage[u][2] + delta_v[0]
        character_storage[u][1] = character_storage[u][1] + delta_v[1]
        character_storage[u][3] = character_storage[u][3] + delta_v[1]
    return character_storage

def find_chars_plate(center_x,center_y, chars):

    a,b = estimate_coef(center_x,center_y)
    centers = np.vstack((center_x, center_y)).T

    uppers = []
    lowers = []
    for center, char in zip(centers, chars):
        if a * center[0] + b - center[1] < 0:
            lowers.append([center[0], char])
        else:
            uppers.append([center[0], char])

    uppers = sorted(uppers, key=lambda x: x[0])
    lowers = sorted(lowers, key=lambda x: x[0])
    string_result = ""
    for upper in uppers:
        string_result += str(upper[1])
    string_result += "-"
    for lower in lowers:
        string_result += str(lower[1])

    return -math.atan(a), string_result


def merge_box(detections):
    dets=[]
    merged_array=[]
    for i in range(len(detections)):
        if i in merged_array:
            continue
        label1, confidence1, box1=detections[i]
        for j in range(i+1,len(detections)):
            label2, confidence2, box2 =detections[j]
            
            box1_=[(int(round(box1[0] - box1[2]/2))), (int(round(box1[1] - box1[3]/2))), (int(round(box1[0] + box1[2]/2))), (box1[1] + box1[3]/2)]
            box2_=[(int(round(box2[0] - box2[2]/2))), (int(round(box2[1] - box2[3]/2))), (int(round(box2[0] + box2[2]/2))), (box2[1] + box2[3]/2)]
            iou=bb_intersection_over_union(np.array(box1_, dtype=np.float32),np.array(box2_, dtype=np.float32))
            # print(iou)
            if  iou> 0.1:
                merged_array.append(j)
                # if j==len(detections)-1:
                #     end_object_flag=True
                label1+="-"+label2
                box1=(min(box1_[0],box2_[0]),min(box1_[1],box2_[1]),max(box1_[2],box2_[2]),max(box1_[3],box2_[3]))
                box1=(float((box1[0]+box1[2])/2),float((box1[1]+box1[3])/2),float((box1[2]-box1[0])),float((box1[3]-box1[1])))
                confidence1=confidence1+"-"+confidence2

        dets.append([label1, confidence1, box1])
    return dets

def merge_box_arr_track(arr_track):
    dets=[]
    merged_array=[]
    # print(arr_track.shape)
    for i in range(arr_track.shape[0]):
        
        if i in merged_array:
            continue
        # print(arr_track[i])
        box1=arr_track[i][:4]
        confidence1,label1=arr_track[i][4:6]
        for j in range(i+1,arr_track.shape[0]):
            box2=arr_track[j][:4]
            confidence2,label2=arr_track[j][4:6]

            iou=bb_intersection_over_union(np.array(list(box1), dtype=np.float32),np.array(list(box2), dtype=np.float32))
            if  iou> 0.1:
                merged_array.append(j)
                label1+=label2
                box1=(min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3]))
                confidence1+=confidence2
        dets.append([box1[0],box1[1], box1[2],box1[3],confidence1,label1])
    arr_track=sorted(dets, key=lambda x: float(x[0]))
    return arr_track