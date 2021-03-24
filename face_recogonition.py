import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

#--- 导入FaceNet相关文件 ---
from pathlib import Path
import sys

from torchvision.transforms.transforms import ToPILImage
FaceNet_dir = Path('src/FaceNet/')
sys.path.append(str(FaceNet_dir))

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from networks.classifier import Classifier

from PIL import Image
import numpy as np

class FaceRecogonition():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.resnet.eval()
        print("--- resnet face loaded ---")
        self.classifer = Classifier(num_classes=3).to(self.device)
        self.classifer.load_state_dict(torch.load("src/FaceNet/checkpoints/classifier_best.pth"))
        self.classifer.eval()
        print("--- face classifier loaded ---")

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])


    def __call__(self, face_img, im_name):
        """
        args: 
            box_img(numpy array) : The person's bounding box
         
        return: 
            face_box
            person_id
        """
        face_img = self.trans(face_img)

        ret = self.resnet(face_img.unsqueeze(0).to(self.device))

        ret = self.classifer(ret)

        _, index = torch.max(ret, dim=1)
        person_id = index.cpu().numpy()


        return person_id




