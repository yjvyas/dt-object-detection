import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import cv2

def save_npz(img, boxes, classes, npz_index):
    np.savez(f"./data_collection/dataset/{npz_index}.npz", img, boxes, classes)

def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    boxes = []
    classes = []
    duckie_mask = (100, 117, 226)
    cone_mask = (226, 111, 101)
    bus_mask = (216, 171, 15)
    truck_mask = (116, 114, 117)
    kernel = np.ones((3,3))
    masks = [{'class':1, 'color':(100, 117, 226)}, 
             {'class':2, 'color':(226, 111, 101)}, 
             {'class':3, 'color':(116, 114, 117)}, 
             {'class':4, 'color':(216, 171, 15)}]

    for m in masks: # used to segment the masks for all objects (excluding background)
        img_masked = cv2.inRange(seg_img, m['color'], m['color'])
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel)
        img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_DILATE, kernel)
        ret, thresh = cv2.threshold(img_masked, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            parent_inds = np.where(hierarchy[0,:,3]==-1)[0].tolist()
            contours = list(map(contours.__getitem__, parent_inds))


            for c in contours:
                classes.append(m['class'])
                x,y,w,h = cv2.boundingRect(c)
                boxes.append([x,y,x+w,y+h])
    
    return np.array(boxes), np.array(classes)

if __name__=='__main__':
    seed(123)
    environment = launch_env()

    policy = PurePursuitPolicy(environment)

    MAX_STEPS = 500

    while True:
        obs = environment.reset()
        environment.render(segment=True)
        rewards = []

        nb_of_steps = 0

        while True:
            action = policy.predict(np.array(obs))

            obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
            segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

            rewards.append(rew)
            environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

            segmented_obs = cv2.resize(segmented_obs, (244,244))
            obs = cv2.resize(obs, (244, 244))

            boxes, classes = clean_segmented_image(segmented_obs)
            # TODO save_npz(obs, boxes, classes)

            nb_of_steps += 1

            if done or nb_of_steps > MAX_STEPS:
                break
