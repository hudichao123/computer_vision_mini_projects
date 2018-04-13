# This Project is about using pretrained ssd model and opencv to do Object Detection
# funny_dog.mp4 and epic-horse.mp4 are input images
# output.mp4 and output_horse.mp4 are output images with bounding boxes indicating the type of detected objects
# ssd.py contains the pretrained single shot detection model that is used in this project

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import random

# Defining a function that will do the detections
def detect(frame,net,transform ): 
    height,width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    #detections = [#batch, # classes, #occurrence, (score,x0,y0,x1,y1)]
    scale = torch.Tensor([width,height,width,height])# create a tensor object of dimensions [width, height, width, height].
    for i in range(detections.size(1)):
        j = 0# initialize the loop variable j that will correspond to the occurrences of the class.
        while(detections[0,i,j,0]>=0.6):# take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            pt = (detections[0, i, j, 1:] * scale).numpy()# get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)# draw a rectangle around the detected object.
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2,cv2.LINE_AA)# put the label of the class right above the rectangle.
            j+=1# increment j to get to the next occurrence.
    return frame# return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd("test")
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) #get the weights of the neural network from another one that is pretrained 

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))#create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.

# Doing some Object Detection on a video
reader = imageio.get_reader("epic-horses.mp4")
fps = reader.get_meta_data()['fps'].
print("fps",fps)
writer = imageio.get_writer("output_horse.mp4",fps = fps)
for (i,frame) in enumerate(reader):
    frame = detect(frame, net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()