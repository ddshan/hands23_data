import os, glob, random, cv2, multiprocessing, argparse
from collections import Counter
from vis_utils import vis_per_image
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



# 272448
# 99012
# 39870
# 76832
# 56734
# count bboxes for masks 796530


def draw_ann(imPath):
    txtPath = imPath+'.txt'
    blurImPath = imPath.replace(hands23, '/w/fouhey/hands2/allMerged7Blur')
    line_ls = read_txt(txtPath)

    # count line by line
    for line in line_ls:
        side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))

        # get image
        im = cv2.imread(blurImPath)
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
    parser.add_argument("--show_vis", action='store_true')
    parser.add_argument("--show_stats", action='store_true')
    parser.add_argument("--table_stats", action='store_true')
    args = parser.parse_args()
    print(args)


    hands23 = '/w/fouhey/hands2/allMerged7'
    hands23_mask_dir = '/x/dandans/hands2/masks_sam_debug'
    save_dir = './vis_ann'
    save_f_dir = './vis_ann_f'
    save_h_dir = './vis_ann_h'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_f_dir, exist_ok=True)
    os.makedirs(save_h_dir, exist_ok=True)

    im_ls = glob.glob(f'{hands23}/*.jpg')
    txt_ls = glob.glob(f'{hands23}/*.txt')

    print(len(im_ls))
    print(len(txt_ls))

    EK_ls = glob.glob(f'{hands23}/EK*.jpg')
    AR_ls = glob.glob(f'{hands23}/AR*.jpg')
    ND_ls = glob.glob(f'{hands23}/ND*.jpg')
    CC_ls = glob.glob(f'{hands23}/CC*.jpg')

    countN = 0
    countSide = []
    countState = []
    countHandBox = []
    countObjBox = []
    countToolType = []
    countSecBox = []
    countGrasp = []
    count_mask = 0

    # plot
    if args.show_vis:
        for name, sub in zip(['ND', 'EK', 'AR', 'CC'], [ND_ls, EK_ls, AR_ls, CC_ls]):
            print(len(sub))
            countN += len(sub)
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

    if args.table_stats:
        print()
        allImage, allHand, allFirst, allSecond, allGrape =0,0,0,0,0
        for name, sub in zip(['ND', 'EK', 'AR', 'CC'], [ND_ls, EK_ls, AR_ls, CC_ls]):
            print(len(sub))
            random.shuffle(sub)
            countSide = []
            countState = []
            countHandBox = []
            countObjBox = []
            countToolType = []
            countSecBox = []
            countGrasp = []

            for imPath in sub:
                txtPath = imPath+'.txt'
                line_ls = read_txt(txtPath)

                # count line by line
                for line in line_ls:
                    side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                    if side != "None":
                        countSide.append(side)
                    if state != "None":
                        countState.append(state)
                    if handBox != "None":
                        countHandBox.append(handBox)
                    if objectBox != "None":
                        countObjBox.append(objectBox)
                    if toolType != "None":
                        countToolType.append(toolType)
                    if secObjectBox != "None":
                        countSecBox.append(secObjectBox)
                    if graspType != "None":
                        countGrasp.append(graspType)

            print(f'sub = {name}')
            print(len(countSide), len(countState), len(countHandBox))
            print(len(countObjBox), len(countToolType))
            print(len(countSecBox))
            print(len(countGrasp))
            print(f'<td>{len(sub)}</td>')
            print(f'<td>{len(countSide)}</td>')
            print(f'<td>{len(countObjBox)}</td>')
            print(f'<td>{len(countSecBox)}</td>')
            print(f'<td>{len(countGrasp)}</td>')
            print()
            allImage += len(sub)
            allHand += len(countHandBox)
            allFirst += len(countObjBox)
            allSecond += len(countSecBox)
            allGrape += len(countGrasp)

        
        print(f'<td>total</td>')
        print(f'<td>{allImage}</td>')
        print(f'<td>{allHand}</td>')
        print(f'<td>{allFirst}</td>')
        print(f'<td>{allSecond}</td>')
        print(f'<td>{allGrape}</td>')
        
            
                

        

    if args.show_stats:
        for name, sub in zip(['ND', 'EK', 'AR', 'CC'], [ND_ls, EK_ls, AR_ls, CC_ls]):
            print(len(sub))
            countN += len(sub)
            random.shuffle(sub)
            sub = sub[:100]
            for imPath in sub:
                txtPath = imPath+'.txt'
                blurImPath = imPath.replace(hands23, '/w/fouhey/hands2/allMerged7Blur')
                # print(txtPath)
                line_ls = read_txt(txtPath)
                if 0:
                    # get image
                    im = cv2.imread(blurImPath)
                    filename = os.path.split(imPath)[-1][:-4] 
                    print(f'cur image = {filename}')

                    # draw
                    annotation = parse_annotation(line_ls)
                    im = vis_per_image(im, annotation, filename, hands23_mask_dir, font_path='./times_b.ttf', use_simple=False)
                    save_path = os.path.join(save_dir, filename+'.png')
                    im.save(save_path)

                # count line by line
                for line in line_ls:
                    side, state, handBox, objectBox, toolType, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                    # countSide.append(side)
                    # countState.append(state)
                    # countHandBox.append(handBox)
                    # countObjBox.append(objectBox)
                    # countToolType.append(toolType)
                    # countSecBox.append(secObjectBox)
                    # countGrasp.append(graspType)

                    if handBox != 'None':
                        count_mask += 1
                    if objectBox != 'None':
                        count_mask += 1
                    if secObjectBox != 'None':
                        count_mask += 1
                

                
                

        print('count masks', count_mask)   


        print(f'totally {countN}')
        print(f'\n Get the distribution:')
        for item in [countSide, countState, countToolType, countGrasp]:
            print(Counter(item))

            # Counter({'right_hand': 230493, 'left_hand': 224321})
            # Counter({'object_contact': 304361, 'no_contact': 105306, 'self_contact': 35171, 'inconclusive': 4998, 'other_person_contact': 4891, 'None': 87})
            # Counter({'None': 173997, 'neither_,_held': 101121, 'neither_,_touched': 81742, 'container_,_held': 36501, 'tool_,_held': 27863, 'tool_,_used': 20587, 'container_,_touched': 12122, 'tool_,_touched': 881})
            # Counter({'None': 389217, 'Pow-Pris': 24706, 'Pre-Pris': 17799, 'NP-Fin': 10640, 'NP-Palm': 5183, 'Pre-Circ': 3386, 'Pow-Circ': 3133, 'Lat': 750})


        



