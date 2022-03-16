import cv2
import glob
import time
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_grpc_detector import YoloV5GRPCDetector

yoloV5GRPCDetector = YoloV5GRPCDetector()

i = 0
start_time = time.time()

glob_path = "../../examples/images/*"

for _ in range(1):
    for img_path in glob.glob(glob_path):
        print(img_path)
        img = cv2.imread(img_path)
        img = img[..., ::-1]
    
        target_boxes = yoloV5GRPCDetector.grpc_detect(img)
        i += 1

print("END", (time.time() - start_time)/i)
