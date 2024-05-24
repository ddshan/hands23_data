# Hands23 data

Data repository for the paper: **Towards A Richer 2D Understanding of Hands at Scale (NeurIPS2023)**.

Tianyi Cheng*, [Dandan Shan*](https://ddshan.github.io/), Ayda Hassen, [Richard Higgins](https://relh.net/), [David Fouhey](https://cs.nyu.edu/~fouhey/).

It contains the SAM mask generation and annotation visulation for Hands23 dataset.

## Download and unzip
```
wget https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/data/hands23_data.zip
unzip hands23_data.zip -d where/to/unzip/
```
After unzip, the folder structure will look like below. `allMergedSplit` contains the train/val/test splits. `allMergedBlur` contains all JPG images. `allMergedTxt` contains all TXT annotation files correspond to JPG images by only adding `.txt` suffix. 
```
hands23_data/
├── allMergedSplit
|   ├── TEST.txt
|   ├── TRAIN.txt
|   └── VAL.txt
├── allMergedBlur
|   └── *.jpg
├── allMergedTxt
|    └── *.jpg.txt
└── masks_sam (to be created in the next step)
    └── *.png

```



## SAM mask generation
The downloaded data `hands23_data.zip` has all other annotations, except for SAM masks. Please follow [Segment Anything](https://github.com/facebookresearch/segment-anything) for the environment installation and downlowd the model checkpoint, the ViT-H SAM model (sam_vit_h_4b8939.pth). Then, you can generate SAM masks for raw format data by simply running
```
python sam/get_sam_masks.py
```




## Visualization
Visualize annotations: load from raw format and plot annotations on images.
```
python vis/vis_hands23.py
```



<!-- ## Data processing
Generate annotation from raw to COCO format.
```
python data_prep/get_coco_format.py
```
 -->


<!-- ## Acknowledgement
SAM -->

## Citing

If you find this data and code useful for your research, please consider citing Hands23 paper,

```bibtex
@inproceedings{cheng2023towards,
        title={Towards a richer 2d understanding of hands at scale},
        author={Cheng, Tianyi and Shan, Dandan and Hassen, Ayda Sultan and Higgins, Richard Ely Locke and Fouhey, David},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023}
        }

```

and also make sure to cite the following paper where the subsets originate: [COCO (Lin et al.)](https://cocodataset.org/#home), [VISOR (Darkhalil et al.)](https://epic-kitchens.github.io/VISOR/) and [Artic. (Qian et al.)](https://jasonqsy.github.io/Articulation3D/).
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

```bibtex
@inproceedings{VISOR,
           title={EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations},
           author={Darkhalil, Ahmad and Shan, Dandan and Zhu, Bin and Ma, Jian and Kar, Amlan and Higgins, Richard and Fidler, Sanja and Fouhey, David and Damen, Dima},
           booktitle   = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
           year      = {2022}
} 
```

```bibtex
@inproceedings{Qian22,
    author = {Shengyi Qian and Linyi Jin and Chris Rockwell and Siyi Chen and David F. Fouhey},
    title = {Understanding 3D Object Articulation in Internet Videos},
    booktitle = {CVPR},
    year = 2022
}
```
