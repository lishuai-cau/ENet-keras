import cv2
import numpy as np
#利用直线变换法进行车道线定位
def line_dect_image(image):
    (rows,cols,channels)=image.shape
    lower_rgb = np.array([0,0,50])
    upper_rgb = np.array([255,255,150])
    image_theshold = cv2.inRange(image,lower_rgb,upper_rgb)

    kernel_width = 5;  # 调试得到的合适的膨胀腐蚀核大小
    kernel_height = 5;  # 调试得到的合适的膨胀腐蚀核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    image_threshold = cv2.erode(image_theshold, kernel)
    image_threshold = cv2.dilate(image_theshold, kernel)
    image_threshold = cv2.dilate(image_theshold, kernel)
    image_threshold = cv2.erode(image_theshold, kernel)
    image_theshold = cv2.morphologyEx(image_theshold, cv2.MORPH_CLOSE, kernel)
    Rho = 1
    Theta = np.pi/90
    CurveNumber = 100
    minLineLength =60
    maxLineGap =10
    lines =cv2.HoughLinesP(image_theshold,Rho,Theta,CurveNumber,minLineLength,maxLineGap)
    try:
        logest_line = 0
        logest_length = 0
        logest_line = lines[0]
        for line in lines:
            for x1,y1,x2,y2 in line:
                lenghth=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
            for x1,y1,x2,y2 in logest_line:
                logest_length=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
            if lenghth>=logest_length:
                logest_line = line
        print("Line detected")
        print("Longes line's length is"+str(logest_length**0.5))
        for x1,y1,x2,y2 in logest_line:
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            cv2.circle(image,(x,y),75,(0,0,255),5)
            cv2.putText(image,str(x),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            # Turn_error = x-cols/2
            Turn_error = cols/2-x
            if Turn_error<30 and Turn_error>-30:
                Turn_error = 0
            linear = 0.2
            angluar = Turn_error*0.0045

    except TypeError or UnboundLocalError:
        print("No line detected")
        linear = 0
        angluar = 0

    # return linear,angluar
    return image,image_theshold