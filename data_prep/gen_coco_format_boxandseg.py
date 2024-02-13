import json, glob, os,shutil, pdb, random, cv2, argparse, shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from data_util import *
random.seed(0)

coco = {
    # "info": {...},
    # "licenses": [...],
    # "images": [...],
    # "annotations": [...],
    # "categories": [...], 
    # "segment_info": [...]
}

handv2_info = {
    "description": "",
    "url": "",
    "version": "1.0",
    "year": 2023,
    "contributor": ""
}

licenses = []

images = [
    # {
    #    "id": 397133,
    #     "file_name": "000000397133.jpg",
    #     "height": 427,
    #     "width": 640
    # }, ...
]



'''
object categories: 
    1:hand
    2:firstobject
    3:secondobject
    
hand features
    side:
        0:left
        1:right
    contact:
        0:incontact
        1:notincontact
    Box:
        [x1, y1, x2, y2]
    segment:
    grasp:
        ...
    handoffset:
        if in contact: [x, y, magnitude]
        otherwise: [-1, -1, -1]

firstobject: 
    Box: 
        [x1, y1, x2, y2]
    toolType:
        ...
    tooloffset:
        [x, y, magnitude]

secondobject:
    Box:
        [x1, y1, x2, y2]
'''


def add_item(image_id=None, 
              category_id=None,
              id=None,
              bbox=None,
              area=None,
              segmentation=None,
              iscrowd=0,
            #   exhaustive=-1,
              handside=None,
              incontact=None,
              handoffset=None,
              grasptype=None,
              tooltype=None,
              tooloffset=None
              ):
    item = {}
    item['id'] = id
    item['image_id'] = image_id
    item['category_id'] = category_id
    #
    item['bbox'] = bbox
    item['area'] = area
    item['segmentation'] = segmentation
    item['iscrowd'] = iscrowd
    
    # additional hand
    # item['exhaustive'] = exhaustive
    # item['handside'] = handside
    # item['isincontact'] = incontact
    # item['offset'] = handoffset
    # item['grasptype'] = grasptype
    
    # additional obj
    # item['tooltype'] = tooltype
    # item['tooloffset'] = tooloffset
    



# ={ 'iscrowd'
#    'bbox': [98, -1, 1127, 718], 
#    'category_id': 1, 

#    'handside': -1, 
#    'isincontact': -1, 
#    'handoffset': [-1, -1, -1], 
#    'grasptype': -1, 
#    'tooltype': -1, 
#    'tooloffset': [-1, -1, -1], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}

    return item



def check_segm_polygon(segm):
    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
    if len(segm) == 0:
        return False, segm
    else:
        return True, segm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy_img', action='store_true', help='Whether to copy image.')
    parser.add_argument('--split', nargs='+', required=True, help='Which split to generate COCO annotations.')
    parser.add_argument('--server', type=str, default='gl')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--use-same', action='store_true')
    args = parser.parse_args()
    
    if args.server == 'epicfail':
        dataRoot = '..'
        # txtBase  = "/w/fouhey/hands2/allMerged6/"
        txtBase = '/w/fouhey/hands2/allMerged7_AllGrasp'
        src      = "/w/fouhey/hands2/allMerged7Blur/"
        maskBase = "/y/ayhassen/allmerged/masks/"
        splitBase = '/w/fouhey/hands2/allMerged7Splits/'

    elif args.server == 'gl':
        dataRoot = "/scratch/fouhey_root/fouhey0/dandans/hands2"
        txtBase  = f"{dataRoot}/allMerged7/"
        src      = f"{dataRoot}/allMerged7Blur/"
        if args.use_sam:
            maskBase = f"{dataRoot}/masks_samV1/"
        else:
            maskBase = f"{dataRoot}/masks/"
        splitBase = f"{dataRoot}/allMerged7Splits/"
    
    
    
    
    target = 'handnew'
    root_dir = f"{dataRoot}/datasets_coco/{target}"
    os.makedirs(root_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test', 'fail', 'hard', 'annotations']:
        folder = f'{root_dir}/{split}'
        os.makedirs(folder, exist_ok=True)
        
    # loop for each image
    allImages = sorted([fn for fn in os.listdir(src) if fn.endswith(".jpg")])
    random.shuffle(allImages)
    print(f'#(total img) = {len(allImages)}')
    
    graspType_ls = []
    toolType_ls  = []
    side_ls      = []
    state_ls     = []
    
    
    # for split in ['fail']:
    # for split in ['train', 'val', 'test']:
    for split in args.split:
        if split == 'hard':
            splitPath = '/x/dandans/workspace/handv2/tmp/HARD.txt'
        elif split == 'fail':
            splitPath = '/x/dandans/workspace/handv2/tmp/FAIL.txt'
        else:
            splitPath = os.path.join(splitBase, split.upper()+'.txt')
        splitContent = open(splitPath).read().strip()
        images = [] if len(splitContent) == 0 else splitContent.split("\n")
        print(f'{split} - {len(images)}')
        
        if args.small:
            images = images[:5000]
        
    
        img_ls, annot_ls = [], []
        img_id, annot_id = 0, 0
        for fn in tqdm(images):
            imagePath = os.path.join(src, fn)
            textPath = os.path.join(txtBase, fn+".txt")
            print(f'img path = {imagePath}')
            
            if args.copy_img:
                newPath = os.path.join(root_dir, split, fn)
                # shutil.copy(imagePath, newPath)
                if not os.path.exists(newPath):
                    os.symlink(imagePath, newPath)
                continue
            
            data = open(textPath).read().strip()
            lines = [] if len(data) == 0 else data.split("\n")
            
            I = cv2.imread(imagePath)
            h, w = I.shape[0], I.shape[1]
            
            img_item = {
                'id': img_id,
                'file_name': fn,
                'height': h,
                'width': w
            }
            
            # copy / softlink image
            if args.copy_img:
                newPath = os.path.join(root_dir, split, fn)
                # shutil.copy(imagePath, newPath)
                os.symlink(imagePath, newPath)
                
                
            # loop for each object in current image
            for lineI, line in enumerate(lines):
                side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                # print(lineI, line)
                
                if graspType not in graspType_ls:  
                    graspType_ls.append(graspType)
                if toolType not in toolType_ls:
                    toolType_ls.append(toolType)
                if side not in side_ls:
                    side_ls.append(side)
                if state not in state_ls:
                    state_ls.append(state)
            
                handCenter = boxStr2Center(handBox, h, w)
                handMask = cv2.imread(os.path.join(maskBase, ("2_%d_" % (lineI))+fn.replace(".jpg",".png")))
                flag = 0
                if handMask is None:
                    flag = 1
                    if fn.find("EK") == -1:
                        print("Missing",fn)
                    handMask = np.zeros((h,w,3))
                    
                handMask2 = (handMask[:, :, 0] > 128).astype(np.uint8)
                handArea, handPolygon = parseMask(handMask2)     
                
                handFlag, handPolygon = check_segm_polygon(handPolygon)
                if handFlag:
                    # hand
                    item = add_item(
                        image_id     = img_id, 
                        category_id  = 1,
                        id           = annot_id,
                        bbox         = boxStr2xywh(handBox, h, w),
                        area         = handArea,
                        segmentation = handPolygon,
                    )
                    annot_ls.append(item)
                    annot_id += 1


                if objectBox != "None":
                    objectCenter = boxStr2Center(objectBox, h, w)
                    objectMask = cv2.imread(os.path.join(maskBase, ("3_%d_" % (lineI))+fn.replace(".jpg",".png")))

                    if objectMask is not None:
                        
                        objectMask = (objectMask[:,:,0] > 128).astype(np.uint8)
                        objectArea, objectPolygon = parseMask(objectMask)
                    
                    else:
                        objectMask = np.zeros((h,w,3))
                        objectMask = (objectMask[:,:,0] > 128).astype(np.uint8)
                        objectArea, objectPolygon = parseMask(objectMask)
                    
                    objectFlag, objectPolygon = check_segm_polygon(objectPolygon)
                    if objectFlag:
                        # hand with first-object 
                        item =  add_item(
                            image_id     = img_id, 
                            category_id  = 2,
                            id           = annot_id,
                            bbox         = boxStr2xywh(objectBox, h, w),
                            area         = objectArea,
                            segmentation = objectPolygon,
                        )
                        annot_ls.append(item)
                        annot_id += 1
                    

                if secObjectBox != 'None':
                    secObjectCenter = boxStr2Center(secObjectBox, h, w)
                    secObjectMask = cv2.imread(os.path.join(maskBase, ("5_%d_" % (lineI))+fn.replace(".jpg",".png")))
                    flag = 0
                    if secObjectMask is not None:
                        secObjectMask = (secObjectMask[:,:,0] > 128).astype(np.uint8)
                        secObjectArea, secObjectPolygon = parseMask(secObjectMask)
                    else:
                        secObjectMask = np.zeros((h,w,3))
                        secObjectMask = (secObjectMask[:,:,0] > 128).astype(np.uint8)
                        secObjectArea, secObjectPolygon = parseMask(secObjectMask)
                        # print(f'secObjectMask is none, secObjectPolygon = {secObjectPolygon}')
                    
                    secObjectFlag, secObjectPolygon = check_segm_polygon(secObjectPolygon)
                    if secObjectFlag:
                        # first-object with second-object
                        item = add_item(
                            image_id     = img_id, 
                            category_id  = 3,
                            id           = annot_id,
                            bbox         = boxStr2xywh(secObjectBox, h, w),
                            area         = secObjectArea,
                            segmentation = secObjectPolygon,
                        )
                        annot_ls.append(item)
                        annot_id += 1
                            
                        
            img_ls.append(img_item)
            img_id += 1
                        
        
        if not args.copy_img:       
            print(f'hand side:{side_ls}')
            print(f'hand state:{state_ls}') 
            print(f'tool type:{toolType_ls}')
            print(f'grasp type:{graspType_ls}')   
            
            
            # assembly
            categories = [
                {"id": 1, "name": "hand"},
                {"id": 2, "name": "firstobject"}, # tool / object
                {"id": 3, "name": "secondobject"} # tool interacted object
            ]
            
            coco['info']         = handv2_info
            coco['licenses']     = licenses
            coco['categories']   = categories  # 0: hand, 1: object
            coco['images']       = img_ls
            coco['annotations']  = annot_ls

            # save
            if args.small:
                f = open(f'{root_dir}/annotations/{split}_small.json', 'w')
            else:
                f = open(f'{root_dir}/annotations/{split}.json', 'w')
            json.dump(coco, f, indent=4, cls=NpEncoder)
            
            # print
            print(f'#image = {len(img_ls)}')
            print(f'#annot = {len(annot_ls)}\n\n')
            print(f'save at {root_dir}/annotations/{split}.json')
            
    
    
        
