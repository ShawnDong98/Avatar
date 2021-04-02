import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np

#--- 导入AlphaPose相关文件 ---
from pathlib import Path
import sys
AlphaPose_dir = Path('./src/AlphaPose/')
sys.path.append(str(AlphaPose_dir))


from opt import opt

from dataloader_webcam import crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import write_json


from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from fn import vis_frame, vis_label


#--- 人脸检测与识别 ---
from face_recogonition import FaceRecogonition


import math
from PIL import Image

class Keypoint(object):
    """关键点"""
    def __init__(self, keypoint_type: int = 0, x_coor: int = 0, y_coor: int = 0, person_id: int = 0,
                toward=None, vision_feature=None):
        """
        Keypoint类的初始化方法。
        :param keypoint_type: (int) 关键点类型编号
        :param x_coor: (int) 高方向的坐标
        :param y_coor: (int) 宽方向的坐标
        :param person_id: (int) 人的编号，默认 0
        :param toward: (boole) 关键点的朝向， True为正面，False为反面，默认 None
        :param vision_feature: (torch.Tensor) 图像特征，默认 None
        """
    
        # 类变量初始化
        self.keypoint_type = keypoint_type
        self.x_coor = x_coor
        self.y_coor = y_coor
        self.person_id = person_id
        if toward is not None:
            self.torward = toward
        if vision_feature is not None:
            self.vision_feature = vision_feature


"""
WebcamLoader: 加载原始数据， 做预处理
"""
class WebcamLoader():
    def __init__(self, webcam):
        if webcam == "0":
            self.stream = cv2.VideoCapture(0)
        else:
            self.stream = cv2.VideoCapture(webcam)

        assert self.stream.isOpened(), 'Cannot capture source'

    def __call__(self, i):
        """
        args: 
            i : 帧数
         
        return: 
            code : 
        """
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []

        (grabbed, frame) = self.stream.read()
        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file

        if not grabbed:
            return None, None, None, None
    
        #--- resize frame to 512 x 256 ---
        frame = cv2.resize(frame, (512, 256))
        
        inp_dim = int(opt.inp_dim)
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
    
        img.append(img_k)
        orig_img.append(orig_img_k)
        im_name.append(str(i)+'.png')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
            
            i = i+1

        return img, orig_img, im_name, im_dim_list



class DetectionLoader():
    def __init__(self):
        self.det_model = Darknet("src/AlphaPose/yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('src/AlphaPose/models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

    def __call__(self, img, orig_img, im_name, im_dim_list):
        """
        args: 
            img : 
            orig_img :    
            im_name :
            im_dim_list : 
         
        return: 
            code : 
        """
        with torch.no_grad():
            # Human Detection
            img = img.cuda()
            prediction = self.det_model(img, CUDA=True)
            dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
            #--- 如果没检测到人 ---
            if isinstance(dets, int) or dets.shape[0] == 0:
                return None, None, None, None, None, None, None
            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            
            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        
        boxes_k = boxes[dets[:,0]==0]
        #--- 如果没检测到人 ---
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return None, None, None, None, None, None, None
        inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)

        return (orig_img[0], im_name[0], boxes_k, scores[dets[:,0]==0], inps, pt1, pt2)



class DetectionProcessor:
    def __init__(self):
        pass

    def __call__(self, orig_img, im_name, boxes, scores, inps, pt1, pt2):
        with torch.no_grad():
            #--- 如果没检测到 ---
            if boxes is None or boxes.nelement() == 0:
                return None, None, None, None, None, None, None
            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)


            return (inps, orig_img, im_name, boxes, scores, pt1, pt2)


class Writer:
    def __init__(self):
        
        self.faceR = FaceRecogonition()

    def __call__(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name, fps_time):
        preds_hm, preds_img, preds_scores = getPrediction(
                    hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

        result = pose_nms(boxes, scores, preds_img, preds_scores)
        result = {
            'imgname': im_name,
            'result': result
        }
        #--- 可视化使用 ---
        img = vis_frame(orig_img, result)
        label, pts_list = vis_label(orig_img, result)

        id_list = []
        for box in boxes:
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
            person_img = orig_img[top:bottom, left:right]
            if person_img.shape[0] >= 5 and person_img.shape[1] >= 5:
                pass
                # person_id = self.faceR(face_img, im_name)
                # id_list.append(person_id)
                # if person_id == 0:
                #     cv2.putText(img, 'clw',(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # if person_id == 1:
                #     cv2.putText(img, 'lwx',(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # if person_id == 2:
                #     cv2.putText(img, 'others',(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # #--- 可视化使用 ---
        # show_img = cv2.putText(orig_img, 'FPS: %f' % (1.0 / (time.time() - fps_time)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show_img = cv2.resize(show_img, (1024, 512))
        # cv2.imshow("AlphaPose Demo", show_img)

        return pts_list, id_list, person_img

class Sk_Detector():
    def __init__(self, batchSize=1):
        args = opt
        args.dataset = 'coco'
        if args.video == '0':
            self.cam = WebcamLoader(0)
        else:
            self.cam = WebcamLoader(args.video)

        self.detection = DetectionLoader()
        self.detection_processor = DetectionProcessor()
        self.writer =  Writer()

        # Load pose model
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()

        self.batchSize = batchSize

    def __call__(self, i, debug=True):
        fps_time = time.time()
        pts_list = None
        id_list = None
        img, orig_img, im_name, im_dim_list = self.cam(i)
        if img is not None:
            orig_img, im_name, boxes_k, scores, inps, pt1, pt2 = self.detection(img, orig_img, im_name, im_dim_list)
        if orig_img is not None:
            inps, orig_img, im_name, boxes, scores, pt1, pt2 = self.detection_processor(orig_img, im_name, boxes_k, scores, inps, pt1, pt2)
        
            if boxes is None or boxes.nelement() == 0:
                return None


            # Pose Estimation
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % self.batchSize:
                leftover = 1
            num_batches = datalen // self.batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*self.batchSize:min((j +  1)*self.batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)


            hm = hm.cpu().data
            pts_list, id_list, person_img = self.writer(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1], fps_time)

        if debug == True:
            return pts_list, id_list, orig_img, person_img
        else:
           return pts_list, id_list

    def KeypointList(self):
        KptList = []
        for i in range(18):
            keypoint = Keypoint(keypoint_type=i, x_coor=0, y_coor=0, person_id=0)
            KptList.append(keypoint)
        
        return KptList

    def pts2keypoints(self, pts_list, id_list):
        person_list = []
        for i in range(len(pts_list)):
            Kpt_List = self.KeypointList()
            for j in range(18):
                Kpt_List[j].keypoint_type = j
                Kpt_List[j].coor_x = pts_list[i][j][0]
                Kpt_List[j].coor_y = pts_list[i][j][1]
                Kpt_List[j].person_id = id_list[i]
            person_list.append(Kpt_List)

        return person_list

    def keypoints2pts(self, person_list):
        pts_list = []
        for i in range(len(person_list)):
            pts = np.zeros((18, 2))
            for j in range(18):
                pts[j][0] = person_list[i][j].coor_x
                pts[j][1] = person_list[i][j].coor_y
            
            pts_list.append(pts)

        return pts_list

    def find_left_right(self, pts):

        left_x,   y = pts.min(axis=0) 
        right_x, y = pts.max(axis=0)

        return left_x, right_x

    def draw_label(self, kp_preds, index=0):

        def process_kps_inverse(kp_preds):
            kp_preds[:, 0] = kp_preds[:, 0] * 2
            kp_preds[:, 1] = kp_preds[:, 1] 

            return kp_preds

        kp_preds = process_kps_inverse(kp_preds)

        left, right = self.find_left_right(kp_preds)

        l_pair = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16), 
        ]




        label = [i for i in range(1, 14)]
        

        img = np.zeros(((256, 512)+(1, ))).astype(np.uint8)
        height,width = img.shape[:2]

        #--- 先缩小分辨率画图，画好后插值上采样 ---
        img = cv2.resize(img,(int(width/2), int(height/2)))
        
        part_line = {}
        for n in range(kp_preds.shape[0]):
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x/2), int(cor_y/2))


            if n == 0:
                img = cv2.circle(img, (int(cor_x/2), int(cor_y/2)), 7, label[12], -1)


        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]


                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (1) + 1
                polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(img, polygon, label[i])

        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_NEAREST)

        # img = Image.fromarray(img)
    

        return img, left, right


def save_datasets(base_dir, orig_img, label, person, f):
    if not os.path.exists(os.path.join(base_dir, "train_img")):
        os.makedirs(os.path.join(base_dir, "train_img"))
    if not os.path.exists(os.path.join(base_dir, "train_label")):
        os.makedirs(os.path.join(base_dir, "train_label"))
    if not os.path.exists(os.path.join(base_dir, "person")):
        os.makedirs(os.path.join(base_dir, "person"))

    print(os.path.join(base_dir, "train_img", f"{f}.png"))
    cv2.imwrite(os.path.join(base_dir, "train_img", f"{f}.png"), orig_img)
    cv2.imwrite(os.path.join(base_dir, "train_label", f"{f}.png"), label)
    cv2.imwrite(os.path.join(base_dir, "person", f"{f}.png"), person)
    


if __name__ == '__main__':
    sk = Sk_Detector()
    f = 0
    while True:
        pts_list, id_list, orig_img, person_img = sk(f)
        label, _, _ = sk.draw_label(pts_list[0])
        cv2.imshow("orig", orig_img)
        cv2.imshow("test", label)
        cv2.imshow("person", person_img)
        save_datasets("../datasets/Avatar/lwx", orig_img, label, person_img, f)
        cv2.waitKey(1)
        f += 1
