import os
import cv2
import time
import torch
import argparse
import numpy as np

from src.AlphaPose.Detection.Utils import ResizePadding
from src.AlphaPose.DetectorLoader import TinyYOLOv3_onecls

from src.AlphaPose.PoseEstimateLoader import SPPE_FastPose
from src.AlphaPose.fn import draw_single, draw_single_test

from src.AlphaPose.Track.Tracker import Detection, Tracker
from src.AlphaPose.ActionsEstLoader import TSSTG

import glob
import math
from PIL import Image

def get_config():
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default="mp.mp4",  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=768,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='448x320',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet101',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='./output',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    return args


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


class PoseDetector():
    def __init__(self, args):
        """ 
        args: 
            args: 初始化模型所需参数
        """
        super(PoseDetector, self)
        self.args = args
        self.device = args.device

        # DETECTION MODEL.
        inp_dets = args.detection_input_size
        self.detect_model = TinyYOLOv3_onecls(inp_dets, device=self.device)

        # POSE MODEL.
        inp_pose = args.pose_input_size.split('x')
        inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
        self.pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=self.device)

        # Tracker.
        max_age = 30
        self.tracker = Tracker(max_age=max_age, n_init=3)

        self.resize_fn = ResizePadding(inp_dets, inp_dets)
        self.f = 0

        self.person_list = []
        for i in range(2):
            KeypointList = []
            for j in range(18):
                keypoint = Keypoint(keypoint_type=j, x_coor=0, y_coor=0, person_id=0)
                KeypointList.append(keypoint)

            self.person_list.append(KeypointList)


        self.pts = np.zeros((14, 2))


    def __call__(self, frame):
        """
        args:
            frame: 摄像头采集的当前帧 -> numpy
        return:
            frame: 当前帧及火柴人的拼接图片 -> numpy
            person_list： 输出是一个二维的list；
                          第一维list表示检测出人的个数；
                          第二维list由14个Keypoint类组成；
                          以单人检测为例，第一维list的维度大小为1（多人则为n）；
                          第二维list维度大小为14， 表示14个关键点。
        """
        self.f += 1
        fps_time = time.time()
        pts = np.zeros((14,3))
        
        frame = cv2.resize(frame, (512, 256))
        orig_img = frame.copy()
        frame = self.preproc(frame)
        bg = (np.ones(frame.shape) * 255).astype(np.uint8)
        bg_ = (np.ones(frame.shape) * 255).astype(np.uint8)
    

        # Detect humans bbox in the frame with detector model.
        detected = self.detect_model.detect(frame, need_resize=False, expand_bb=10)
        # if detected is not None:
            # print("detected: ", detected.shape)
        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        self.tracker.predict()
        # Merge two source of predicted bbox together.
        for track in self.tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det
            

        detections = []  # List of Detections object for tracking.
        # if detected is None:
        #     self.clear_person_list()
        if detected is not None:
            
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = self.pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(self.kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if self.args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        self.tracker.update(detections)

        person_list = []
        # Predict Actions of each track.
        for i, track in enumerate(self.tracker.tracks):
            # print(i)
            if not track.is_confirmed():
                continue

            # VISUALIZE.
            if track.time_since_update == 0:
                if self.args.show_skeleton:
                    # print(track.keypoints_list[-1][0])
                    frame, pts = draw_single(frame, track.keypoints_list[-1])
                    pts = self.pts_to_512_256(pts)
                    # bg_, _, _ = self.draw_label(pts)
                    
                    person = self.KeypointList()
                    for j in range(18):
                        if j < 1 :
                            person[j].keypoint_type = j
                            person[j].x_coor = pts[j][0]
                            person[j].y_coor = pts[j][1]
                            person[j].person_id = i

                        if j==1 or j==2 or j==3 or j==4:
                            person[j].keypoint_type = j
                            person[j].x_coor = 0
                            person[j].y_coor = 0
                            person[j].person_id = i
                        
                        if j > 4:
                            person[j].keypoint_type = j
                            person[j].x_coor = pts[j-4][0]
                            person[j].y_coor = pts[j-4][1]
                            person[j].person_id = i

                    person_list.append(person)


                    

        # Show Frame.
        # frame = cv2.resize(frame, (512, 256))
        frame = cv2.putText(frame, '%d, FPS: %f' % (self.f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame = frame[:, :, ::-1]

        return orig_img, person_list

    def KeypointList(self):
        KptList = []
        for i in range(18):
            keypoint = Keypoint(keypoint_type=i, x_coor=0, y_coor=0, person_id=0)
            KptList.append(keypoint)
        
        return KptList

    def clear_person_list(self):
        for i in range(2):
            for j in range(18):
                if j < 1 :
                    person[j].keypoint_type = j
                    person[j].x_coor = 0
                    person[j].y_coor = 0
                    person[j].person_id = i

                if j==1 or j==2 or j==3 or j==4:
                    person[j].keypoint_type = j
                    person[j].x_coor = 0
                    person[j].y_coor = 0
                    person[j].person_id = i
                
                if j > 4:
                    person[j].keypoint_type = j
                    person[j].x_coor = 0
                    person[j].y_coor = 0
                    person[j].person_id = i

    def find_left_right(self, pts):

        left_x, y, _ = pts.min(axis=0) 
        right_x, y, _ = pts.max(axis=0)

        return left_x, right_x
    
    def kpt2bbox(self, kpt, ex=20):
        """Get bbox that hold on all of the keypoints (x,y)
        kpt: array of shape `(N, 2)`,
        ex: (int) expand bounding box,
        """
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                        kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

    def preproc(self, image):
        """preprocess function for CameraLoader.
        """
        image = self.resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def pts_to_256(self, pts):
        pts = pts * 256 / self.args.detection_input_size
        return pts

    def pts_to_512_256(self, pts):
        pts[:, 0] = (pts[:, 0] * 512 / 768 / 2).clip(0, 256)
        pts[:, 1] = ((pts[:, 1] - 192) * 256 / 384).clip(0, 256)

        return pts

    def list2pts(self, person_list):
        for i in range(18):
            if i < 1:
                self.pts[i][0] = person_list[0][i].x_coor
                self.pts[i][1] = person_list[0][i].y_coor 
            if i > 4:
                self.pts[i-4][0] = person_list[0][i].x_coor
                self.pts[i-4][1] = person_list[0][i].y_coor 

        return self.pts

    def draw_label(self, kp_preds, index=0):

        def process_kps_inverse(kp_preds):
            kp_preds[:, 0] = kp_preds[:, 0] * 2
            kp_preds[:, 1] = kp_preds[:, 1] 

            return kp_preds

        kp_preds = process_kps_inverse(kp_preds)

        left, right = self.find_left_right(kp_preds)

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


if __name__ == '__main__':
    args = get_config()

    cam_source = args.camera
    if cam_source == '0':
        cam_source = 0
    cam = cv2.VideoCapture(cam_source)

    detector = PoseDetector(args)



    while True:
        ret, frame = cam.read()

        if ret:
            frame, person_list = detector(frame)

            # # 渲染
            # out = render(person_list)
            
            

        else:
            break

        # 
        cv2.imshow('frame', frame)

        # # 渲染
        # cv2.imshow('test', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()