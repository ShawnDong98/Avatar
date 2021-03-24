"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from collections import OrderedDict

#--- 导入SPADE相关文件 ---
from pathlib import Path
import sys
SPADE_dir = Path('./src/SPADE/')
sys.path.append(str(SPADE_dir))

import data
from options_1.test_options import TestOptions as TestOptions1
from options_2.test_options import TestOptions as TestOptions2
from models.pix2pix_model import Pix2PixModel
import util.util as util
from util.visualizer import Visualizer
from util import html

import cv2
import numpy as np
import time
import math


from data.base_dataset import BaseDataset, get_params, get_transform
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


class Reciever():
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
            kp_preds[:, 0] = kp_preds[:, 0] * 4
            kp_preds[:, 1] = kp_preds[:, 1] * 2

            return kp_preds

        kp_preds = process_kps_inverse(kp_preds)

        left, right = self.find_left_right(kp_preds)

        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16), 
        ]


        label = [i for i in range(1, 16)]


        img = np.zeros(((512, 1024)+(1, ))).astype(np.uint8)
        height,width = img.shape[:2]

        #--- 先缩小分辨率画图，画好后插值上采样 ---
        img = cv2.resize(img,(int(width/2), int(height/2)))
        
        part_line = {}
        for n in range(kp_preds.shape[0]):
            #--- 不管概率多少，全部画出来 ---
            # if kp_scores[n] <= 0.05:
            #     continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x/2), int(cor_y/2))

            #--- 不画下面的关节点 ---
            # bg = img.copy()
            # cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
            # # Now create a mask of logo and create its inverse mask also
            # transparency = float(max(0, min(1, kp_scores[n])))
            # img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)

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
                stickwidth = (2) + 1
                polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(img, polygon, label[i])

        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_NEAREST)

        img = Image.fromarray(img)
    

        return img, left, right


class Render_SPADE():
    def __init__(self):
        self.opt1 = TestOptions1().parse()
        self.opt2 = TestOptions2().parse()
        self.model1 = Pix2PixModel(self.opt1)
        self.model1.eval()

        self.model2 = Pix2PixModel(self.opt2)
        self.model2.eval()

        self.reciever = Reciever()
        

    def render(self, label, index=1):
        params = get_params(self.opt1, label.size)
        transform_label = get_transform(self.opt1, params, method=Image.NEAREST, normalize=False)

        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt1.label_nc  # 'unknown' is opt.label_nc

        data_i = {'label': label_tensor.unsqueeze(0),
                      'instance': torch.Tensor(0),
                      'image': torch.Tensor(0),
                      'path': '',
                      }

        if index == 1:
            generated = self.model1(data_i, mode='inference')
        else:
            generated = self.model2(data_i, mode='inference')

        # start_time = time.time()
        img = util.tensor2im(generated.squeeze(0))[:, :, ::-1]
        label = util.tensor2label(data_i['label'].squeeze(0), self.opt1.label_nc)



        return img, label


    def __call__(self, person_list):
        if len(person_list) > 1:
            fps_time = time.time()
            pts_list = self.reciever.keypoints2pts(person_list)
            frame1, left1, right1 = self.reciever.draw_label(pts_list[0])
            frame2, left2, right2 = self.reciever.draw_label(pts_list[1])

    

            img1, label1 = self.render(frame1, index=1)
            img2, label2 = self.render(frame2, index=2)

            if(left2 > right1):
                cut_line = int(right1 + (left2 - right1) / 2)
                img = np.concatenate((img1[:, :cut_line, :], img2[:, cut_line:, :]), axis=1)
                label = np.concatenate((label1[:, :cut_line, :], label2[:, cut_line:, :]), axis=1)
            else:
                cut_line = int(right2 + (left1 - right2) / 2)
                img = np.concatenate(( img2[:, :cut_line, :], img1[:, cut_line:, :]), axis=1)
                label = np.concatenate((label2[:, :cut_line, :], label1[:, cut_line:, :]), axis=1)

            out = np.concatenate((label, img), axis=1)

            out1 = out.copy()
            # #--- 可视化使用 ---
            # out1 = cv2.putText(out1, 'FPS: %f' % (1.0 / (time.time() - fps_time)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return out1


        else:
            fps_time = time.time()
            pts_list = self.reciever.keypoints2pts(person_list)
            label, left, right = self.reciever.draw_label(pts_list[0])
            img, label = self.render(label, index=2)
            
            out = np.concatenate((label, img), axis=1)
            out1 = out.copy()
            # #--- 可视化使用 ---
            # out1 = cv2.putText(out1, 'FPS: %f' % (1.0 / (time.time() - fps_time)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return out1


if __name__ == '__main__':

    from glob import glob

    labels = sorted(glob("./datasets/clw_30//train_label*.png"))
    render = Render()

    for label in labels:
        print(label)
        label = Image.open(label)
        start_time = time.time()
        out = render(label)
        time_cost = time.time() - start_time
        
        print("fps: ", 1 / time_cost)

        cv2.imshow("test", out)
        cv2.waitKey(1)





