import os
from collections import OrderedDict
from torch.autograd import Variable

#--- 导入pix2pixHD相关文件 ---
from pathlib import Path
import sys
pix2pixHD_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixHD_dir))

from options_1.test_options import TestOptions as TestOptions1
from options_2.test_options import TestOptions as TestOptions2
from options_3.test_options import TestOptions as TestOptions3
from options_4.test_options import TestOptions as TestOptions4
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize

from PIL import Image
import numpy as np
import cv2
import math
import time


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
                Kpt_List[j].x_coor = pts_list[i][j][0]
                Kpt_List[j].y_coor = pts_list[i][j][1]
                Kpt_List[j].person_id = i
            person_list.append(Kpt_List)

        return person_list

    def keypoints2pts(self, person_list):
        pts_list = []
        for i in range(len(person_list)):
            pts = np.zeros((18, 2))
            for j in range(18):
                pts[j][0] = person_list[i][j].x_coor
                pts[j][1] = person_list[i][j].y_coor
            
            pts_list.append(pts)

        return pts_list

    def find_left_right(self, pts):

        left_x,   y = pts.min(axis=0) 
        right_x, y = pts.max(axis=0)

        return left_x, right_x

   
    def draw_label_18(self, kp_preds, index=0):

        def process_kps_inverse(kp_preds):
            kp_preds[:, 0] = kp_preds[:, 0] * 2
            kp_preds[:, 1] = kp_preds[:, 1]

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


        img = np.zeros(((256, 512)+(1, ))).astype(np.uint8)
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
                stickwidth = (1) + 1
                polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(img, polygon, label[i])

        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_NEAREST)

        img = Image.fromarray(img)
    

        return img, left, right

    def draw_label(self, kp_preds, index=0):
        
        pts = np.zeros((14, 2))
        def process_kps_inverse(kp_preds):
            kp_preds[:, 0] = kp_preds[:, 0] * 2
            kp_preds[:, 1] = kp_preds[:, 1] 

            return kp_preds

        kp_preds = process_kps_inverse(kp_preds)

        #
        pts[0, :] = kp_preds[0, :]
        pts[1:, :] = kp_preds[5:, :]

        left, right = self.find_left_right(pts)

        l_pair = [
            (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
            (13, 7), (13, 8),  # Body
            (7, 9), (8, 10), (9, 11), (10, 12), 
        ]




        label = [i for i in range(1, 14)]
        

        img = np.zeros(((256, 512)+(1, ))).astype(np.uint8)
        height,width = img.shape[:2]

        #--- 先缩小分辨率画图，画好后插值上采样 ---
        img = cv2.resize(img,(int(width/2), int(height/2)))
        
        part_line = {}
        for n in range(pts.shape[0]):
            cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
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

        img = Image.fromarray(img)
    

        return img, left, right



class Render_pix2pixHD():
    def __init__(self):
        self.opt1 = TestOptions1().parse(save=False)
        self.opt2 = TestOptions2().parse(save=False)
        self.opt3 = TestOptions3().parse(save=False)
        self.opt4 = TestOptions4().parse(save=False)
        self.opt1.nThreads = 1   # test code only supports nThreads = 1
        self.opt1.batchSize = 1  # test code only supports batchSize = 1
        self.opt1.serial_batches = True  # no shuffle
        self.opt1.no_flip = True  # no flip

        # test
        if not self.opt1.engine and not self.opt1.onnx:
            self.model1 = create_model(self.opt1)
            self.model2 = create_model(self.opt2)
            self.model3 = create_model(self.opt3)
            self.model4 = create_model(self.opt4)
            if self.opt1.data_type == 16:
                self.model1.half()
                self.model2.half()
                self.model3.half()
                self.model4.half()
            elif self.opt1.data_type == 8:
                self.model1.type(torch.uint8)
                self.model2.type(torch.uint8)
                self.model3.type(torch.uint8)
                self.model4.type(torch.uint8)
                    
            if self.opt1.verbose:
                # print(model)
                pass
        else:
            from run_engine import run_trt_engine, run_onnx


        self.reciever = Reciever()

        self.last_person1_id = 0
        self.last_person2_id = 1


        self.bg = np.ones((256, 512, 3), dtype=np.uint8)

    def render(self, A, index=1):
        params = get_params(self.opt1, A.size)
        transform_A = get_transform(self.opt1, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = torch.Tensor(0)

        data = {'label': A_tensor.unsqueeze(0), 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': ''}

        if self.opt1.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif self.opt1.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if self.opt1.export_onnx:
            print ("Exporting to ONNX: ", self.opt1.export_onnx)
            assert self.opt1.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(self.model, [data['label'], data['inst']],
                            self.opt1.export_onnx, verbose=True)
            exit(0)

        minibatch = 1 
        if self.opt1.engine:
            generated = run_trt_engine(self.opt1.engine, minibatch, [data['label'], data['inst']])
        elif self.opt1.onnx:
            generated = run_onnx(self.opt1.onnx, self.opt1.data_type, minibatch, [data['label'], data['inst']])
        else:   
            if index == 1:     
                generated = self.model1.inference(data['label'], data['inst'], data['image'])
            if index == 2:
                generated = self.model2.inference(data['label'], data['inst'], data['image'])
            if index == 3:     
                generated = self.model3.inference(data['label'], data['inst'], data['image'])
            if index == 4:
                generated = self.model4.inference(data['label'], data['inst'], data['image'])

        img = util.tensor2im(generated.data[0])

        img = img[:, :, ::-1]

        return img

    def __call__(self, person_list):
        #--- 如果画面中没有人 ---
        if len(person_list) == 0:
            self.bg[:, :, :] = (0, 255, 0)
            return self.change_bg(self.bg)

        pts_list = self.reciever.keypoints2pts(person_list)
        
        if len(person_list) > 1:
            fps_time = time.time()
            
            frame1, left1, right1 = self.reciever.draw_label(pts_list[0])
            frame2, left2, right2 = self.reciever.draw_label(pts_list[1])

            if person_list[0][0].person_id is None or person_list[0][0].person_id >= 4:
                person_list[0][0].person_id = self.last_person1_id

            if person_list[1][0].person_id is None or person_list[1][0].person_id >= 4:
                person_list[1][0].person_id = self.last_person2_id


            img1 = self.render(frame1, index=person_list[0][0].person_id + 1)
            img1 = self.change_bg(img1)
            img2 = self.render(frame2, index=person_list[1][0].person_id + 1)
            img2 = self.change_bg(img2)

            self.last_person1_id = person_list[0][0].person_id
            self.last_person2_id = person_list[1][0].person_id

            if(left2 > right1):
                cut_line = int(right1 + (left2 - right1) / 2)
                img = np.concatenate((img1[:, :cut_line, :], img2[:, cut_line:, :]), axis=1)
            else:
                cut_line = int(right2 + (left1 - right2) / 2)
                img = np.concatenate(( img2[:, :cut_line, :], img1[:, cut_line:, :]), axis=1)

            out = img


            return out


        else:
            fps_time = time.time()
            label, left, right = self.reciever.draw_label(pts_list[0])

            if person_list[0][0].person_id is None or person_list[0][0].person_id >= 4:
                person_list[0][0].person_id = self.last_person1_id

            img = self.render(label, index=person_list[0][0].person_id + 1)
            img = self.change_bg(img)

            self.last_person1_id = person_list[0][0].person_id
            
            out = img
    

            return out

    def if_no_person(self, pts_list):
        if(pts_list[0].any() == 0 and pts_list[1].any() == 0):
            return True


    def change_bg(self, frame, index=0):
        height, width, channel = frame.shape
        bg = np.zeros((height, width, channel))
        if index == 0:
            bg[:, :, 0] = 255
            bg[:, :, 1] = 0
            bg[:, :, 2] = 0
        elif index == 1:
            bg[:, :, 0] = 0
            bg[:, :, 1] = 255
            bg[:, :, 2] = 0
        elif index == 2:
            bg[:, :, 0] = 0
            bg[:, :, 1] = 0
            bg[:, :, 2] = 255
        elif index == 3:
            bg[:, :, 0] = 255
            bg[:, :, 1] = 255
            bg[:, :, 2] = 0
        elif index == 4:
            bg[:, :, 0] = 0
            bg[:, :, 1] = 255
            bg[:, :, 2] = 255
        else:
            bg[:, :, 0] = 255
            bg[:, :, 1] = 0
            bg[:, :, 2] = 255

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#进行色彩值转换，RGB到HSV
        lower_hsv = np.array([35,43,46])#色彩范围h s v三变量的最小取值
        upper_hsv = np.array([77, 255, 255])#色彩范围h s v三变量的最小取值
        mask=cv2.inRange(hsv,lowerb=lower_hsv,upperb=upper_hsv)  #进行色值去范围，取出对应的色彩范围进行过滤

        mask_bg = mask / 255.
        mask_fg=cv2.bitwise_not(mask) / 255.

        frame[:, :, 0] = frame[:, :, 0] * mask_fg + bg[:, :, 0] * mask_bg 
        frame[:, :, 1] = frame[:, :, 1] * mask_fg + bg[:, :, 1] * mask_bg 
        frame[:, :, 2] = frame[:, :, 2] * mask_fg + bg[:, :, 2] * mask_bg 

        return frame



if __name__ == "__main__":

    label = Image.open("0.png")

    render = Render_pix2pixHD()

    out = render(label)


    cv2.imshow("out", out)

    cv2.waitKey(0)