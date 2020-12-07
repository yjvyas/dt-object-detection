import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask, img_boxes
from data_collection import save_npz, clean_segmented_image
import cv2
import os

def scale_boxes(boxes, classes, original_dim, new_dim):
    new_boxes = []
    new_classes = []
    for i, b in enumerate(boxes):
        b_new = [round(b[0]/original_dim[0]*new_dim[0]),
                 round(b[1]/original_dim[1]*new_dim[1]),
                 round(b[2]/original_dim[0]*new_dim[0]),
                 round(b[3]/original_dim[1]*new_dim[1])]
        if (b_new[2]-b_new[0])*(b_new[3]-b_new[1]) > 20: # remove boxes that are too small
            new_boxes.append(b_new)
            new_classes.append(classes[i])
    return new_boxes, new_classes

if __name__=='__main__':
    npz_index = 0

    seed(123)
    environment = launch_env()

    policy = PurePursuitPolicy(environment)

    dataset_size = 2000
    MAX_STEPS = 100
    save_every = 20

    img_dir = "./data_collection/dataset/train_images/"
    os.makedirs(img_dir, exist_ok=True)

    while npz_index < dataset_size:
        obs = environment.reset()

        nb_of_steps = 0

        while True:
            action = policy.predict(np.array(obs))

            obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
            segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

            # segmented_obs = cv2.resize(segmented_obs, (244,244))
            obs = cv2.resize(obs, (244, 244))

            boxes, classes = clean_segmented_image(segmented_obs)

            if len(boxes)==0:
                break # sometimes it's just road
            boxes, classes = scale_boxes(boxes, classes, [640, 480], [244, 244])

            img_annotated = img_boxes(obs, boxes, classes)

            img_name = img_dir + "img_{}.jpg".format(npz_index)

            if nb_of_steps == 0 or nb_of_steps % save_every == 0:
                cv2.imwrite(img_name, cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
                print("Saved image {}.".format(img_name))

            save_npz(obs, boxes, classes, npz_index)
            print("Saved npz file with index {}.".format(npz_index))
            nb_of_steps += 1
            npz_index += 1

            if done or nb_of_steps > MAX_STEPS or npz_index > dataset_size:
                break
