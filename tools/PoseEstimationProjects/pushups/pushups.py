import mediapipe as mp
import cv2
import numpy as np

def get_keypoint(results, height, width):
    left_shoulder_x = int(results.pose_landmarks.landmark[11].x * width)
    left_shoulder_y = int(results.pose_landmarks.landmark[11].y * height)
    left_shoulder_xy = [left_shoulder_x, left_shoulder_y]

    left_elbow_x = int(results.pose_landmarks.landmark[13].x * width)
    left_elbow_y = int(results.pose_landmarks.landmark[13].y * height)
    left_elbow_xy = [left_elbow_x, left_elbow_y]

    left_wrist_x = int(results.pose_landmarks.landmark[15].x * width)
    left_wrist_y = int(results.pose_landmarks.landmark[15].y * height)
    left_wrist_xy = [left_wrist_x, left_wrist_y]

    left_hip_x = int(results.pose_landmarks.landmark[23].x * width)
    left_hip_y = int(results.pose_landmarks.landmark[23].y * height)
    left_hip_xy = [left_hip_x, left_hip_y]

    left_knee_x = int(results.pose_landmarks.landmark[25].x * width)
    left_knee_y = int(results.pose_landmarks.landmark[25].y * height)
    left_knee_xy = [left_knee_x, left_knee_y]

    left_ankle_x = int(results.pose_landmarks.landmark[27].x * width)
    left_ankle_y = int(results.pose_landmarks.landmark[27].y * height)
    left_ankle_xy = [left_ankle_x, left_ankle_y]

    return left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy




def calc_distance(x1, y1, x2, y2, x3, y3):
    u = np.array([x2 - x1, y2 - y1])
    v = np.array([x3 - x1, y3 - y1])
    L = abs(np.cross(u, v)/np.linalg.norm(u))
    return L



def calc_slope(left_shoulder_xy, left_ankle_xy):
    x = [left_shoulder_xy[0], left_ankle_xy[0]]
    y = [left_shoulder_xy[1], left_ankle_xy[1]]
    slope, intercept = np.polyfit(x,y,1)
    return slope


def get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, flg_low, slope, dist_hip, dist_knee, dist_elbow):
    if slope <= THRESH_SLOPE and dist_hip < THRESH_DIST_SPINE and \
                    dist_knee < THRESH_DIST_SPINE and dist_elbow > THRESH_ARM:
        flg_low = True
    else:
        flg_low = False
    return flg_low



def draw_keypoint(image, RADIUS, CLR_KP, CLR_LINE, THICKNESS, left_shoulder_xy, \
                  left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy):

    cv2.circle(image, (left_shoulder_xy[0], left_shoulder_xy[1]), RADIUS, CLR_KP, THICKNESS)
    cv2.circle(image, (left_elbow_xy[0], left_elbow_xy[1]), RADIUS, CLR_KP, THICKNESS)
    cv2.circle(image, (left_wrist_xy[0], left_wrist_xy[1]), RADIUS, CLR_KP, THICKNESS)
    cv2.circle(image, (left_hip_xy[0], left_hip_xy[1]), RADIUS, CLR_KP, THICKNESS)
    cv2.circle(image, (left_knee_xy[0], left_knee_xy[1]), RADIUS, CLR_KP, THICKNESS)
    cv2.circle(image, (left_ankle_xy[0], left_ankle_xy[1]), RADIUS, CLR_KP, THICKNESS)

    cv2.line(image, (left_shoulder_xy[0], left_shoulder_xy[1]), (left_elbow_xy[0], left_elbow_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_elbow_xy[0], left_elbow_xy[1]), (left_wrist_xy[0], left_wrist_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_shoulder_xy[0], left_shoulder_xy[1]), (left_hip_xy[0], left_hip_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_hip_xy[0], left_hip_xy[1]), (left_knee_xy[0], left_knee_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_knee_xy[0], left_knee_xy[1]), (left_ankle_xy[0], left_ankle_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)

    return image


if __name__ == "__main__":

    THRESH_SLOPE = 0
    THRESH_DIST_SPINE = 30
    THRESH_ARM = 40

    RADIUS = 5
    THICKNESS = 2
    CLR_KP = (0, 0, 255)
    CLR_LINE = (255, 255, 255)


    mp_pose = mp.solutions.pose

    cap_file = cv2.VideoCapture('training.mp4')

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:

        count = 0
        flg_low = False

        while cap_file.isOpened:
            success, image = cap_file.read()
            if not success:
                print("empty camera frame")
                break
            image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height = rgb_image.shape[0]
            width = rgb_image.shape[1]

            results = pose_detection.process(rgb_image)

            if not results.pose_landmarks:
                print('not results')
            else:
                left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy = get_keypoint(results, height, width)

                dist_hip = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_hip_xy[0], left_hip_xy[1])

                dist_knee = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_knee_xy[0], left_knee_xy[1])

                dist_elbow = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_wrist_xy[0], left_wrist_xy[1], left_elbow_xy[0], left_elbow_xy[1])
    
                body_slope = calc_slope(left_shoulder_xy, left_ankle_xy)


                pre_flg_low = flg_low
                flg_low = get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, \
                                       flg_low, body_slope, dist_hip, dist_knee, dist_elbow)

                if pre_flg_low == False and flg_low == True:
                    count += 1

                cv2.putText(image, str(int(count)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                image = draw_keypoint(image, RADIUS, CLR_KP, CLR_LINE, THICKNESS, \
                                    left_shoulder_xy, left_elbow_xy, left_wrist_xy, \
                                    left_hip_xy, left_knee_xy, left_ankle_xy)



            cv2.imshow('Pushup counter test', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break


    print('合計'+str(count)+'回')

cap_file.release()
cv2.destroyAllWindows()
