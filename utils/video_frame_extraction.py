import cv2
import os
from cv2 import CAP_PROP_FPS

'''
frames_per_second: How many frames to be captured each second
num_frames: Total amount of frames to capture
'''
def video_to_frames(video_path, output_path, frames_per_second=None, num_frames=None, crop=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(CAP_PROP_FPS)
    capture_frequency = 1
    if frames_per_second is not None:
      capture_frequency = int(round(fps / frames_per_second, 0))

    index = 0        
    while cap.isOpened() and (num_frames is None or index < (capture_frequency * num_frames)):
        Ret, Mat = cap.read()
        if Ret:
            if index % capture_frequency == 0:
                if crop: cv2.imwrite(output_path + '/' + str(index) + '.png', Mat[100:984, 940:1824])
                else: cv2.imwrite(output_path + '/' + output_path.split('/')[-1] + '_' + str(index) + '.png', Mat)
            index += 1
        else:
            break
    cap.release()
    return

def multiple_videos_to_frames(folder_path, output_path, frames_per_second=None, num_frames=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(folder_path)))
    print(videos)
    
    for video in videos:
        print('\nProcessing video "' + video + '"...\n')
        video_to_frames(folder_path + '/' + video, output_path + '/' + video[:3], frames_per_second, num_frames)
    