# Framework

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
    cd segmentation
    bash tools/dist_train.sh configs/pspnet_onns/pspnet_wres38-d8_10k_histo.py 1 runs/onns
    
    (inference, patch merge and evaluation)
    bash tools/dist_test.sh configs/pspnet_onns/pspnet_wres38-d8_10k_histo_test.py [path to best checkpoint] 1
    python tools/merge_patches.py luad/test_patches luad/test_merged 2
    python tools/count_miou.py luad/test_merged [path to original val gt] 2



