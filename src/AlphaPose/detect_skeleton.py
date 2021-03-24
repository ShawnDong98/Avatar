import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, Writer, crop_from_dets, Mscoco
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


class sk_detector():
    def __init__(self):
        args = opt
        args.dataset = 'coco'
        webcam = args.video
        mode = args.mode
        if not os.path.exists(args.outputpath):
            os.mkdir(args.outputpath)

        # Load input video
        data_loader = WebcamLoader(args.video).start()
        (fourcc,fps,frameSize) = data_loader.videoinfo()

        # Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
        self.det_processor = DetectionProcessor(det_loader).start()


        # Load pose model
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()

        # Data writer
        save_path = os.path.join(args.outputpath, 'AlphaPose_webcam'+webcam+'.avi')
        self.writer = Writer(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize)



        self.batchSize = args.posebatch

        

    def __call__(self):
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = self.det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                self.writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                return False

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
            self.writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            pts_list, im_name = self.writer.update()

            for i in range(len(pts_list)):
                if not os.path.exists(f"./output/{opt.video.split('.')[-2]}/{opt.video.split('.')[-2]}_pts/"):
                    os.makedirs(f"./output/{opt.video.split('.')[-2]}/{opt.video.split('.')[-2]}_pts/")
                print(f"./output/{opt.video.split('.')[-2]}/{opt.video.split('.')[-2]}_pts/{im_name.split('.')[-2]}.txt")
                np.savetxt(f"./output/{opt.video.split('.')[-2]}/{opt.video.split('.')[-2]}_pts/{im_name.split('.')[-2]}.txt", pts_list[i])

        return pts_list

    def KeypointList(self):
        KptList = []
        for i in range(18):
            keypoint = Keypoint(keypoint_type=i, x_coor=0, y_coor=0, person_id=0)
            KptList.append(keypoint)
        
        return KptList

    def pts2keypoints(self, pts_list):
        person_list = []
        for i in range(len(pts_list)):
            Kpt_List = self.KeypointList()
            for j in range(18):
                Kpt_List[j].keypoint_type = j
                Kpt_List[j].coor_x = pts_list[i][j][0]
                Kpt_List[j].coor_y = pts_list[i][j][1]
                Kpt_List[j].person_id = i
            person_list.append(Kpt_List)

        return person_list

    def keypoints2pts(self, person_list):
        pts_list = []
        for i in range(len(person_list)):
            pts = np.zeros(18, 2)
            for j in range(18):
                pts[j][0] = person_list[i][j].coor_x
                pts[j][1] = person_list[i][j].coor_y
            
            pts_list.append(pts)

        return pts_list





if __name__ == "__main__":
    detector = sk_detector()
    while True:
        pts_list = detector()
        person_list = detector.pts2keypoints(pts_list)
        print("person_list[0].person_id: ", person_list[0][0].person_id)
        # print("person_list[1].person_id: ", person_list[1][0].person_id)

    

    
    
    

    
