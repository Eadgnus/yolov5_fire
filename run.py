from PIL import ImageGrab
import cv2
import mouse
import numpy as np
import os
import detect
import ultralytics


def set_roi():
    global ROI_SET, x1, y1, x2, y2
    ROI_SET = False
    print("Select your ROI using mouse drag.")
    while(mouse.is_pressed() == False):
        x1, y1 = mouse.get_position()
        while(mouse.is_pressed() == True):
            x2, y2 = mouse.get_position()
            while(mouse.is_pressed() == False):
                print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
                run()
                return


x1, y1, x2, y2 = 0, 0, 0, 0
def run():
    while True:
            print('loop 시작')
            # 템포러리 파일 경로 설정
            temp_file = 'temp.jpg'

            # 현재 프레임 캡처 및 이미지 저장
            image = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 이미지 파일로 저장
            cv2.imwrite(temp_file, image)
            # 객체 감지 수행
            detect.run(weights='./best.pt', source=temp_file, imgsz=(x2-x1, y2-y1), conf_thres=0.25, exist_ok='./output/', view_img=True)
            key = cv2.waitKey(100)
            if key == ord("q"):
                print("Quit")
                break
            # 템포러리 파일 삭제
            os.remove(temp_file)
    cv2.destroyAllWindows()


set_roi()