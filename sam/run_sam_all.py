import os, glob, random, cv2, pdb, imageio, json
import numpy as np
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

random.seed(0)

def decode_video(video_path, save_dir):
    video_name = os.path.split(video_path)[-1][:-4]
    cmd = f"ffmpeg -i {video_path} '{save_dir}/{video_name}_%06d.jpg'"
    # cmd = f"ffmpeg -i {video_path} %06d.png"
    os.system(cmd)
    print(f"{video_path}, #frames={len(os.listdir(save_dir))}")

def create_video(video_path, save_dir):
    # cmd = f"ffmpeg -framerate 10 -pattern_type glob -i '{save_dir}/*.jpg' -c:v -pix_fmt yuv420p {video_path}"
    # os.system(cmd)
    writer = imageio.get_writer(video_path, fps=10)
    im_ls = glob.glob(f'{save_dir}/*.jpg')
    im_ls.sort()
    for file in im_ls:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()


def plot_mask(image, masks, save_path=None):
    mask_image = np.ones_like(image) * np.array([30, 144, 255]).astype(np.uint8)
    gray = np.stack((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),)*3, axis=-1)
    blend = cv2.addWeighted(gray, 0.5, mask_image.astype(np.uint8), 0.5, 0.0)
    for mask in masks:
        mask = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(blend, contours, -1, (255,255,255), 2)
    
    
    
    res = np.concatenate((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), blend), axis=1)
    cv2.imwrite(save_path, res)

def write_masks_to_folder(masks, save_dir, image=None, video_name=None) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'_crop', exist_ok=True)
    os.makedirs(save_dir+'_crop_mask', exist_ok=True)
    data = {}
    tmp = None
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(save_dir, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
        x, y, w, h = mask_data['bbox']

        mask = mask.astype(np.uint8)
        im = (image * mask[:, :, np.newaxis])[:, :, ::-1]
        im = np.concatenate((im, mask[:, :, np.newaxis] * 255), axis=-1)
        crop = im[y:y+h, x:x+w]
        crop_mask = mask[y:y+h, x:x+w] * 255

        image_folder = (save_dir+'_crop')
        mask_folder = (save_dir+'_crop_mask')
        # cv2.imwrite(os.path.join(save_dir+'_crop', filename), crop[:, :, ::-1])
        cv2.imwrite(os.path.join(image_folder, f"{i}_{x}_{y}.png"), crop)
        cv2.imwrite(os.path.join(mask_folder, f"{i}_{x}_{y}.png"), crop_mask)
        
        image_file_path ='./' + image_folder + f"/{i}_{x}_{y}.png"
        mask_file_path  ='./' + mask_folder + f"/{i}_{x}_{y}.png"
        data[f'{i}']={
            'image_path': image_file_path,
            'mask_path': mask_file_path,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }
        if i == 0: tmp = im[:, :, :-1]
        else: tmp += im[:, :, :-1]
    print(data)

    cv2.imwrite( f"000.png", tmp)

    metadata_path = os.path.join(save_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    with open(video_name+'.json', "w") as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':

    '''python run_sam_all.py
    '''
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    video_dir = f'/nfs/turbo/fouheyTemp/dandans/datasets/short_videos/short_videos_2'
    vis_dir  = f'vis_puzzle'
    vis_video_dir = f'{vis_dir}/videos_plot'
    os.makedirs(vis_video_dir, exist_ok=True)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam.to(device=device)

    video_ls = glob.glob(f'{video_dir}/*/*.mp4')
    random.shuffle(video_ls)


    for video_path in tqdm(video_ls[:200]): 
        video_name = os.path.split(video_path)[-1]
        frame_dir = vis_dir+f'/{video_name[:-4]}'
        os.makedirs(frame_dir, exist_ok=True)
        decode_video(video_path, frame_dir)

        # for image_path in glob.glob(frame_dir+f'/*.jpg'):
        for image_path in ['kitchen.jpg']:
            video_name = 'kitchen'
            mask_dir = f'{vis_dir}/{video_name}'
            mask_path = f'{mask_dir}/{os.path.split(image_path)[-1]}'
            os.makedirs(mask_dir, exist_ok=True)
            if os.path.exists(image_path):
                print(image_path)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                masks = mask_generator.generate(image)
                # save vis
                plot_mask(image, masks, mask_path)
                # save meta
                write_masks_to_folder(masks, save_dir=mask_path[:-4], image=image, video_name=video_name)
                # pdb.set_trace()
            exit(0)

        create_video(f'{vis_video_dir}/{video_name}', mask_dir)
            



