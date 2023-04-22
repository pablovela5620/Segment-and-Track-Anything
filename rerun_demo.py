import argparse
import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc

import rerun as rr

io_args = {
    'input_video': './assets/cell.mp4',
    'output_mask_dir': './assets/cell_masks', # save pred masks
    'output_video': './assets/cell_seg.mp4', # mask+frame vizualization
    'output_gif': './assets/cell_seg.gif', # mask visualization
}

# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}

def track():
    # source video to segment
    cap = cv2.VideoCapture(io_args['input_video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    # output masks
    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = segtracker_args['sam_gap']
    frame_idx = 0
    segtracker = SegTracker(segtracker_args,sam_args,aot_args)
    segtracker.restart_tracker()


    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            rr.log_image("image/rgb", frame)
            if frame_idx == 0:
                pred_mask = segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                segtracker.add_reference(frame, pred_mask)
            elif (frame_idx % sam_gap) == 0:
                seg_mask = segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = segtracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)
                # save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                segtracker.add_reference(frame, pred_mask)
            else:
                pred_mask = segtracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            rr.log_segmentation_image("image/pred_mask", pred_mask)
            
            # save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')
            # # masked_frame = draw_mask(frame,pred_mask)
            # # masked_pred_list.append(masked_frame)
            # # plt.imshow(masked_frame)
            # # plt.show() 
            
            # pred_list.append(pred_mask)
            
            
            print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
            frame_idx += 1
        cap.release()
        print('\nfinished')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IDEA Research Grounded Dino + SAM example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "grounded_sam")
    track()
    rr.script_teardown(args)


if __name__ == "__main__":
    main()