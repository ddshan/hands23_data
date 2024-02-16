import os, glob, random, cv2, multiprocessing, argparse, shutil
from collections import Counter
from vis_utils import vis_per_image
from tqdm import tqdm
random.seed(0)


def read_txt(path):
    line_ls = open(path, 'r').readlines()
    line_ls = [x.strip() for x in line_ls]
    return line_ls


def parse_annotation(line_ls):
    im_ann = []
    for line in line_ls:
        side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
        hand_ann = {
            "hand_side": side,
            "contact_state": state,
            "hand_bbox":[float(x) for x in handBox.split(',')] if handBox !="None" else None,

            "obj_bbox": [float(x) for x in objectBox.split(',')] if objectBox !="None" else None,
            "obj_touch": toolType,

            "second_obj_bbox": [float(x) for x in secObjectBox.split(',')] if secObjectBox !="None" else None,
            "grasp": graspType
        }
        im_ann.append(hand_ann)
    return im_ann



def draw_ann(imPath):
    txtPath = imPath+'.txt'
    line_ls = read_txt(txtPath)

    # count line by line
    for line in line_ls:
        side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))

        # get image
        im = cv2.imread(imPath)
        filename = os.path.split(imPath)[-1][:-4] 
        print(f'cur image = {filename}')

        # draw
        annotation = parse_annotation(line_ls)
        im, im_f, im_h = vis_per_image(im, annotation, filename, hands23_mask_dir, font_path='./times_b.ttf', use_simple=False)
        save_path = os.path.join(save_dir, filename+'.png')
        im.save(save_path)

        save_path = os.path.join(save_dir+'_f', filename+'.png')
        im_f.save(save_path)

        save_path = os.path.join(save_dir+'_h', filename+'.png')
        im_h.save(save_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--copy_data", action='store_true')
    parser.add_argument("--show_vis", action='store_true')
    parser.add_argument("--show_stats", action='store_true')
    parser.add_argument("--table_stats", action='store_true')
    args = parser.parse_args()
    print(args)

    hands23_root   = '/x/dandans/hands23_data_release'
    hands23_split  = hands23_new_root + f'/allMergedSplit'
    hands23_blur   = hands23_new_root + f'/allMergedBlur'
    hands23_txt    = hands23_new_root + f'/allMergedTxt'
    hands23_mask   = hands23_new_root + f"/masks_sam'

    
    EK_ls = glob.glob(f'{hands23_blur}/EK*.jpg')
    AR_ls = glob.glob(f'{hands23_blur}/AR*.jpg')
    ND_ls = glob.glob(f'{hands23_blur}/ND*.jpg')
    CC_ls = glob.glob(f'{hands23_blur}/CC*.jpg')


    save_dir   = './vis_ann'
    save_f_dir = './vis_ann_f'
    save_h_dir = './vis_ann_h'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_f_dir, exist_ok=True)
    os.makedirs(save_h_dir, exist_ok=True)


    for name, sub in zip(['ND', 'EK', 'AR', 'CC'], [ND_ls, EK_ls, AR_ls, CC_ls]):
        print(len(sub))
        random.shuffle(sub)
        select_ls = []
        for imPath in sub:
            txtPath = imPath+'.txt'
            line_ls = read_txt(txtPath)
            for line in line_ls:
                side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                if secObjectBox != 'None':
                    select_ls.append(imPath)
                    continue
            if len(select_ls) > 100:
                break
            
        P = multiprocessing.Pool(36)
        P.map(draw_ann, select_ls)   


