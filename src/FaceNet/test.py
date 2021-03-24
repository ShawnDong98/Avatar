import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image

from networks.classifier import Classifier


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=160, margin=14, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(
    pretrained='vggface2',
).to(device)

resnet.eval()

classifer = Classifier().to(device)
classifer.load_state_dict(torch.load("./checkpoints/classifier_best.pth"))
classifer.eval()

trans = transforms.Resize((512, 512))
unload = transforms.ToPILImage()


img = Image.open("35.png")
img = trans(img)
aligned = mtcnn(img)

print(aligned.shape)
ret = resnet(aligned.unsqueeze(0).to(device))

ret = classifer(ret)

_, index = torch.max(ret, dim=1)

print(index)