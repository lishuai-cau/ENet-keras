import copy
import os
import random
import cv2
import time
import numpy as np
from PIL import Image
from nets.ENet import enet
from line_follower import line_dect_image

#---------------------------------------------------#
#   定义了输入图片的颜色，当我们想要去区分两类的时候
#   我们定义了两个颜色，分别用于背景和赛道线
#   [0,0,0], [0,255,0]代表了颜色的RGB色彩
#---------------------------------------------------#
class_colors = [[0,0,0],[128,0,0]]
#---------------------------------------------#
#   定义输入图片的高和宽，以及种类数量
#---------------------------------------------#
HEIGHT = 320
WIDTH = 320
#---------------------------------------------#
#   背景 + 赛道线 = 2
#---------------------------------------------#
NCLASSES = 2

#---------------------------------------------#
#   载入模型
#---------------------------------------------#
# model = convnet_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model = enet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
#--------------------------------------------------#
#   载入权重，训练好的权重会保存在logs文件夹里面
#   我们需要将对应的权重载入
#   修改model_path，将其对应我们训练好的权重即可
#   下面只是一个示例
#--------------------------------------------------#
model_path = "./logs/ep064-loss0.006-val_loss0.007.h5"
model.load_weights(model_path)

# model.save("ENet.h5")
def predict(img):
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]
    img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)
    img = np.array(img) / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    pr = model.predict(img)[0]
    pr = pr.reshape(HEIGHT, WIDTH, NCLASSES).argmax(axis=-1)
    seg_img = np.zeros((HEIGHT, WIDTH, 3))
    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:,: ] == c) * class_colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((pr[:,: ] == c) * class_colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((pr[:,: ] == c) * class_colors[c][2]).astype('uint8')
    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    return seg_img

capture = cv2.VideoCapture("out.mp4")

fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    if ref == True:
        frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.imshow("origin", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(predict(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video",frame)
        image,image_theshold = line_dect_image(frame)
        cv2.imshow("image", image)
        cv2.imshow("theshold",image_theshold)
        # out.write(frame)
    else:
        break


    c= cv2.waitKey(30) & 0xff
    if c==27:
        capture.release()
        break
