import cv2
import json
import os.path
import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch2trt
from torch2trt import TRTModule
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from IPython import embed



def get_keypoint(outputs, w, h, threshold=0.2):
    body_part_ids = [5, 6, 7, 11, 12, 13]
    keypoints = []
    detected = False
    for key in outputs:
        if key > 0:
            detected = True
            break
        
    if detected:
        for i in body_part_ids:
            conf_map = outputs[i]
            try:
                max_conf_idx = np.argmax(conf_map)
                y, x = np.unravel_index(max_conf_idx, conf_map.shape)
                if conf_map[y, x] < threshold:
                    keypoints.append((None, None))
                else:
                    keypoints.append((x * w, y * h))
            except ValueError:
                keypoints.append((None, None))
        return keypoints
    else:
        return None





def calc_distance(x1, y1, x2, y2, x3, y3):
    if x1 is None or y1 is None or x2 is None or y2 is None or x3 is None or y3 is None:
        return None
    u = np.array([x2 - x1, y2 - y1])
    v = np.array([x3 - x1, y3 - y1])
    L = abs(np.cross(u, v) / np.linalg.norm(u))
    return L


def calc_slope(left_shoulder_xy, left_ankle_xy):
    if left_shoulder_xy is None or left_ankle_xy is None:
        return None
    x = [left_shoulder_xy[0], left_ankle_xy[0]]
    y = [left_shoulder_xy[1], left_ankle_xy[1]]
    if None in x or None in y:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return slope



def get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, flg_low, slope, dist_hip, dist_knee, dist_elbow):
    if slope is None or dist_hip is None or dist_knee is None or dist_elbow is None:
        return None
    if slope <= THRESH_SLOPE and dist_hip < THRESH_DIST_SPINE and \
            dist_knee < THRESH_DIST_SPINE and dist_elbow > THRESH_ARM:
        flg_low = True
    else:
        flg_low = False
    return flg_low


# def draw_keypoint(image, RADIUS, CLR_KP, CLR_LINE, THICKNESS, keypoints):
#     connections = [(0, 1), (1, 2), (5, 6), (11, 12), (12, 13)]
#     if keypoints:
#         for i, j in connections:
#             p1 = keypoints[i]
#             p2 = keypoints[j]
#             if p1[0] is not None and p2[0] is not None:
#                 cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), CLR_LINE, THICKNESS, lineType=cv2.LINE_AA, shift=0)
#         for k in keypoints:
#             if k[0] is not None:
#                 cv2.circle(image, (int(k[0]), int(k[1])), RADIUS, CLR_KP, THICKNESS)
#     return image


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

    
def main():
    WIDTH = 224
    HEIGHT = 224
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    
    print("loading human_pose.json...")
    with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    print("loaded human_pose.json")
    
    
    #if optimized model does not exist
    if not os.path.isfile(OPTIMIZED_MODEL):
        print('optimizing weight...')
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda()
    
        WEIGHTS_PATH = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        model.load_state_dict(torch.load(WEIGHTS_PATH))

        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        #optimizing model
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        #saving optimized model
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
        print('optimizing done')

    #loading the optimized model
    print('loading model..')
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))  
    print('loaded model')
    
    
    coco_category_id = trt_pose.coco.coco_category_to_parts(human_pose)
    
    #setting up csi camera
    print('setting up camera...')
    cap = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)
    cap.running = True
    print('camera set')

    # cap = cv2.VideoCapture(0)
    
    RADIUS = 5
    THICKNESS = 2
    CLR_KP = (0, 0, 255)
    CLR_LINE = (255, 255, 255)

    THRESH_SLOPE = 0
    THRESH_DIST_SPINE = 30
    THRESH_ARM = 40

    count = 0
    flg_low = False
    pre_flg_low = False

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    
    while True:
        #Inference
        image = cap.value
        data = preprocess(image)
        with torch.no_grad():
            cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(image, counts, objects, peaks)
        
        keypoints = None
        for i in range(counts[0]):
            keypoints = get_keypoint(objects[0][i], WIDTH, HEIGHT)
            break
        
        if keypoints is not None and None not in keypoints:
            left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy = keypoints
            dist_hip = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_hip_xy[0], left_hip_xy[1])
            dist_knee = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_knee_xy[0], left_knee_xy[1])
            dist_elbow = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_wrist_xy[0], left_wrist_xy[1], left_elbow_xy[0], left_elbow_xy[1])
            body_slope = calc_slope(left_shoulder_xy, left_ankle_xy)

            pre_flg_low = flg_low
            flg_low = get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, flg_low, body_slope, dist_hip, dist_knee, dist_elbow)
            if flg_low == None:
                continue
            elif pre_flg_low == False and flg_low == True:
                count += 1

        frame = cv2.UMat(np.array(image))
        cv2.imshow('detected_pose',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        print(count)
        
    
if __name__ == '__main__':
    main()