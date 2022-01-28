import cv2
import glob
import time
from NomeroffNet.YoloV5GRPCDetector import YoloV5GRPCDetector

yoloV5GRPCDetector = YoloV5GRPCDetector()

i = 0
start_time = time.time()

glob_path = "../../examples/images/*"

for _ in range(1):
    for img_path in glob.glob(glob_path):
        print(img_path)
        img = cv2.imread(img_path)
        img = img[..., ::-1]
    
        targetBoxes = yoloV5GRPCDetector.grpc_detect(img)
        i += 1

print("END", (time.time() - start_time)/i)
