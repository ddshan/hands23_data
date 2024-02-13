'''Run SAM on our new dataset.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2, pdb
from tqdm import tqdm
import random

random.seed(10)

def show_mask(image, mask, bbox, save_path, random_color=False):
    if random_color:
        # color = np.concatenate([np.random.randint(low=0, hight=255, size=(3)), np.array([0.6])], axis=0)
        color = np.random.randint(low=0, hight=255, size=(3)).astype(np.uint8)
    else:
        color = np.array([30, 144, 255]).astype(np.uint8)
        color_bbox = (255, 255, 255)
    h, w = mask.shape[-2:]

    mask_image = np.zeros_like(image)
    for i in range(mask.shape[0]):
        ma = mask[i].astype(np.uint8)
        mask_image += ma.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # print(f'mask_image shape = {mask_image.shape}')
        # print(f'image shape = {image.shape}')

    gray = np.stack((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),)*3, axis=-1)
    # blend = gray * ( 1 - mask_image) * 0.5 + image * mask_image * 0.5
    blend = cv2.addWeighted(gray, 0.5, mask_image.astype(np.uint8), 0.5, 0.0)
    
    if bbox is not None:
        for i in range(bbox.shape[0]):
            bb = bbox[i]
            blend = cv2.rectangle(blend, (bb[0], bb[1]), (bb[2], bb[3]), color=color_bbox, thickness=3)
    
    res = np.concatenate((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), blend), axis=1)
    cv2.imwrite(save_path, res)
    # pdb.set_trace()





    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



if __name__ == '__main__':

    

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    import sys, os, random
    from tqdm import tqdm
    sys.path.append("..")
    from data_prep.data_util import *
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.utils.transforms import ResizeLongestSide
    

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    def prepare_image(image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device) 
        return image.permute(2, 0, 1).contiguous()
    

    predictor = SamPredictor(sam)

    # start on our dataset
    txtBase  = "/w/fouhey/hands2/allMerged7/"
    src      = "/w/fouhey/hands2/allMerged7Blur"
    maskBase = "/y/ayhassen/allmerged/masks/"
    splitBase = '/w/fouhey/hands2/allMerged7Splits/'
    

    target = 'handnew_sam'
    root_dir = f"/x/dandans/workspace/handv2/sam/vis/{target}"
    os.makedirs(root_dir, exist_ok=True)

    # loop for each image
    allImages = sorted([fn for fn in os.listdir(src) if fn.endswith(".jpg")])
    random.shuffle(allImages)
    print(f'#(total img) = {len(allImages)}')

    count_bad = 0

    for split in ['train']: #, 'val', 'test']:
        splitPath = os.path.join(splitBase, split.upper()+'.txt')
        splitContent = open(splitPath).read().strip()
        images = [] if len(splitContent) == 0 else splitContent.split("\n")
        print(f'{split} - {len(images)}')
        random.shuffle(images)

        batched_input = []
        batched_image = []
        batched_bbox  = []
        batched_imagepath = []
        for fn in tqdm(images):
            imagePath = os.path.join(src, fn)
            if not os.path.exists(imagePath): continue
            textPath = os.path.join(txtBase, fn+".txt")
            
            data = open(textPath).read().strip()
            lines = [] if len(data) == 0 else data.split("\n")

            image = cv2.imread(imagePath)
            h, w = image.shape[0], image.shape[1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            bbox_ls = []
            print()
            
            for lineI, line in enumerate(lines):
                side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                print(lineI, line)
            
                # single prompt
                # input_box = np.array(boxStr2xyxy(handBox, h, w))
                # masks, _, _ = predictor.predict(
                #     point_coords=None,
                #     point_labels=None,
                #     box=input_box[None, :],
                #     multimask_output=False,
                # )
                # show_mask(image, masks, input_box[np.newaxis,:], f'{root_dir}/{fn}.jpg')


                # (2) batched prompt
                bbox_ls.append(boxStr2xyxy(handBox, h, w))
                if objectBox  != 'None':
                    bbox_ls.append(boxStr2xyxy(objectBox, h, w))
                if secObjectBox != 'None':
                    bbox_ls.append(boxStr2xyxy(secObjectBox, h, w))

            
            input_boxes = torch.tensor(bbox_ls, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            # try:
            if len(bbox_ls) == 0: continue
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,)
            print('masks shape', masks.shape)
            print('input boxes shape', input_boxes.shape)    
            masks = masks.cpu().numpy()
            # except:
            #     count_bad += 1
            #     print(f'bad image: {count_bad}')
            #     continue

            show_mask(image, masks, np.array(bbox_ls), f'{root_dir}/{fn}.jpg')
            
            
            # (3) batched end2end
        #     cur = {
        #             'image': prepare_image(image, resize_transform, sam),
        #             'boxes': resize_transform.apply_boxes_torch(input_boxes, image.shape[:2]),
        #             'original_size': image.shape[:2]
        #         }
        #     batched_input.append(cur)
        #     batched_image.append(image)
        #     batched_bbox.append(bbox_ls)
        #     batched_imagepath.append(fn)

        # batched_output = sam(batched_input, multimask_output=False)
        # for i, o in tqdm(enumerate(batched_output)):
        #     masks = o['masks']
        #     image = batched_image[i]
        #     bbox_ls = batched_bbox[i]
        #     name = batched_imagepath[i]
        #     show_mask(image, masks, np.array(bbox_ls), f'{root_dir}/{fn}.jpg')



        # exit(0)

