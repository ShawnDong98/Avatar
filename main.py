from render_pix2pixHD import Render_pix2pixHD
from skeleton_detector import get_config, PoseDetector

import cv2



args = get_config()

cam_source = args.camera
if cam_source == '0':
    cam_source = 0
cam = cv2.VideoCapture(cam_source)

detector = PoseDetector(args)
render = Render_pix2pixHD()


while True:
    ret, frame = cam.read()

    if ret:
        frame, person_list = detector(frame)

        print(len(person_list))

        # 渲染
        out = render(person_list)
        
        

    else:
        break

    # 
    cv2.imshow('frame', frame)

    # 渲染
    cv2.imshow('test', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()