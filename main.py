from render_pix2pixHD import Render_pix2pixHD
# from render_SPADE import Render_SPADE
from detect_skeleton import Sk_Detector

import cv2

import time



if __name__ == '__main__':
    detector = Sk_Detector()
    render = Render_pix2pixHD()
    # render = Render_SPADE()
    f = 0
    while True:
        fps_time = time.time()

        ret = detector(f)

        if ret is None:
            continue
        # print("type(ret[0]: ", type(ret[0]))
        # print("type(ret[1]: ", type(ret[1]))
        pts_list, id_list = ret
        
        if pts_list is None:
            continue

        person_list = detector.pts2keypoints(pts_list, id_list)


        if len(person_list) > 2 or len(person_list) == 0:
            continue

        out = render(person_list)



        out = cv2.resize(out,(1024, 512), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("out", out)
        cv2.waitKey(1)

        f += 1
