import cv2

cap = cv2.VideoCapture("./data/task_sample_videos/jump_forward.mp4")
c = 1
frameRate = 1  # 帧数截取间隔（每隔2帧截取一帧）

while (True):
    ret, frame = cap.read()
    if ret:
        if (c % frameRate == 0):
            print("Start cropping：" + str(c) + " frame")
            # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
            cv2.imwrite("./data/task04/v_JumpForward_g01/frame" + str(c) + '.png', frame)  # 这里是将截取的图像保存在本地
        c += 1
        cv2.waitKey(0)
    else:
        print("All frames are saved")
        break

cap.release()