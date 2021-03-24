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


from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from fn import vis_frame, vis_label


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
        im_name.append(str(i)+'.jpg')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
            
            i = i+1

        return img, orig_img, im_name, im_dim_list



class DetectionLoader():
    def __init__(self):
        self.det_model = Darknet("./src/AlphaPose/yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('./src/AlphaPose/models/yolo/yolov3-spp.weights')
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
        
        pass

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

        #--- 可视化使用 ---
        img = cv2.putText(img, 'FPS: %f' % (1.0 / (time.time() - fps_time)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("AlphaPose Demo", img)

        return pts_list, 

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

    def __call__(self, i):
        fps_time = time.time()
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
            self.writer(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1], fps_time)


if __name__ == '__main__':
    sk = Sk_Detector()
    f = 0
    while True:
        sk(f)
        cv2.waitKey(1)
        f += 1
