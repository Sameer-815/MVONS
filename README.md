# Weakly Supervised Histopathology Tissue Semantic Segmentation with Multi-scale Voting and Online Noise Suppression
This repository contains the official code for the paper:([Link)](https://www.sciencedirect.com/science/article/pii/S0952197625011017)
<details>
<summary>Read full abstract.</summary>
The development of an Artificial Intelligence (AI) assisted tissue segmentation method of digital pathology images is critical for cancer diagnosis and prognosis. Excellent performance has been achieved with the current fully supervised segmentation approach, which relies on a huge number of annotated data. However, drawing dense pixel-level annotations on the giga-pixel whole slide image (WSI) is extremely time-consuming and labor-intensive. To this end, we propose a tissue segmentation method using only patch-level classification labels to reduce such annotation burden and significantly improve the quality of the pseudo-masks. We introduce a framework with two phases of classification and segmentation. In the classification phase, we propose a multi-scale voting method on the Class Activation Map (CAM) based model to obtain more stable pseudo masks. In the segmentation phase, an Online Noise Suppression Strategy (ONSS) is proposed to encourage the model to focus on more reliable signals in the pseudo mask rather than noisy signals. Extensive experiments on two weakly supervised pathology image tissue segmentation datasets Lung Adenocarcinoma (LUAD-HistoSeg) and Breast Cancer Semantic Segmentation (BCSS-WSSS) demonstrate our model outperforms state-of-the-art weakly-supervised semantic segmentation (WSSS) methods using patch-level labels. Furthermore, our method exhibits superior generalization ability compared to other models, and demonstrates promising adaptation performance on unseen domains with only small amounts of data.
</details>

# Overview

![framework](framework.png)



# Weakly-supervised Classification part

## Pretrained weights and datasets

Download the pretained weight of classification stage via Google Cloud Drive ([Link)](https://drive.google.com/file/d/1Rka2SzqAwxUEFb28tbmiy2anhkkFOnTg/view?usp=drive_link)

Download the datasets via Google Cloud Drive ([Link)](https://drive.google.com/file/d/1lWAeCp6UN30VRVmqv97kA2sJ1Pp2frhC/view?usp=drive_link)([Link)](https://drive.google.com/file/d/178eSM9xs5jITt5P2kjaswDlJzwlU5gps/view?usp=drive_link)

## Training classification model and generate pesudo masks

1、Train the classification model:

```python
cd 1.classfication
python 1_train_stage1.py
```

2、generate pesudo masks with the image-level label:

```python
python 2_generate_PM.py
```

# Weakly-supervised Semantic Segmentation part
    
    (train)
    cd 2.segmentation
    bash tools/dist_train.sh configs/pspnet_onns/pspnet_wres38-d8_10k_histo.py 1 runs/onns
    
    (inference, patch merge and evaluation)
    bash tools/dist_test.sh configs/pspnet_onns/pspnet_wres38-d8_10k_histo_test.py [path to best checkpoint] 1
    python tools/merge_patches.py luad/test_patches luad/test_merged 2
    python tools/count_miou.py luad/test_merged [path to original val gt] 2
    
# Contact
If you have any question, please contact us at [sameervim99@gmail.com](mailto:sameervim99@gmail.com)

# Citing

If you find this useful, please cite:
```bibtex
@article{pan2025weakly,
  title={Weakly supervised histopathology tissue semantic segmentation with multi-scale voting and online noise suppression},
  author={Pan, Xipeng and Zhang, Hualong and Deng, Huahu and Wang, Huadeng and Li, Lingqiao and Liu, Zhenbing and Wang, Lin and An, Yajun and Lu, Cheng and Liu, Zaiyi and others},
  journal={Engineering Applications of Artificial Intelligence},
  volume={156},
  pages={111100},
  year={2025},
  publisher={Elsevier}
}
```

