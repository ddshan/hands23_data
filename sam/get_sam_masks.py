'''Run SAM on hands23 dataset.
'''
import sys, os, random, cv2, pdb
sys.path.append('../')
import numpy as np
import torch, os, argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_util import *
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def save_mask(mask, save_path):
    p, t = os.path.split(save_path)
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    assert mask.shape[0] <=3, f'More than 3 masks'

    for i in range(mask.shape[0]):
        m = mask[i]
        m = m.reshape(h, w, 1)
        
        if i == 0:
            path = f'{p}/2_{t}'
            handmask = m
        elif i == 1:
            path = f'{p}/3_{t}'
            m = m - np.logical_and(handmask>0, m>0)
        elif i == 2:
            path = f'{p}/5_{t}'
            m = m - np.logical_and(handmask>0, m>0)
        else:
            print(f'Out of index')
            pdb.set_trace()

        ma = np.concatenate((m, m, m), axis=2) * 255

        cv2.imwrite(path, ma)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', nargs='+', required=True, help='Which split to generate COCO annotations.')
    parser.add_argument('--hands23_root', type=str, default='/path/to/hands23_data', help='Which dateset to generate SAM labels.')
    args = parser.parse_args()

    # define SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    predictor = SamPredictor(sam)
    
    # data dir
    hands23_root = args.hands23_root
    txtBase   = f"{hands23_root}/allMergedTxt"
    src       = f"{hands23_root}/allMergedBlur"
    splitBase = f"{hands23_root}/allMergedSplit"
    
    mask_dir = f'{hands23_root}/masks_sam'
    os.makedirs(mask_dir, exist_ok=True)

    
    for split in args.split: #['train', 'val', 'test']:
        
        splitPath = os.path.join(splitBase, split.upper()+'.txt')
        print(f'split file = {splitPath}')
        splitContent = open(splitPath).read().strip()
        images = [] if len(splitContent) == 0 else splitContent.split("\n")
        print(f'{split} - {len(images)}')

        batched_input = []
        batched_image = []
        batched_bbox  = []
        batched_imagepath = []
        for fn in tqdm(images):
         
            imagePath = os.path.join(src, fn)
            textPath = os.path.join(txtBase, fn+".txt")
            if not os.path.exists(imagePath): 
                print(f'image not exist: {imagePath}')
                breakpoint()
            if not os.path.exists(textPath): 
                print(f'txt not exist: {textPath}')
                breakpoint()

            data = open(textPath).read().strip()
            lines = [] if len(data) == 0 else data.split("\n")

            image = cv2.imread(imagePath)
            h, w = image.shape[0], image.shape[1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            for lineI, line in enumerate(lines):
                bbox_ls = []
                side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))

                # prep batched bbox prompt
                bbox_ls.append(boxStr2xyxy(handBox, h, w))
                if objectBox  != 'None':
                    bbox_ls.append(boxStr2xyxy(objectBox, h, w))
                if secObjectBox != 'None':
                    bbox_ls.append(boxStr2xyxy(secObjectBox, h, w))
                if len(bbox_ls) == 0: continue

                input_boxes = torch.tensor(bbox_ls, device=predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

                # run SAM             
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,)

                # save masks
                masks = masks.cpu().numpy()
                save_mask(masks, save_path=f'{mask_dir}/{lineI}_{fn[:-4]}.png')
            
