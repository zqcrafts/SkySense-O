<div align="center">
<head>
  <style>
    .centered-content {
      text-align: center;  /* 居中对齐 */
    }
    .image {
      margin-bottom: -40px; /* 图片与下方内容的间距 */
    }
  </style>
</head>
<body>
  <div class="centered-content">
    <img class="image" width="600" alt="image" src="docs/figures/icon.png">
    <br>
  </div>
</body>

<div style="position: relative; text-align: center;">
  <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 100%; height: 0; border-top: 4px dashed gray; z-index: 10;"></div>
</div>

<!-- <div style="position: relative; text-align: center;"> <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 100%; height: 4px; background-color: gray; z-index: 10;"></div> </div>  -->



<!-- <h3>✨ In CVPR 2025 <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a>  ✨</h3> -->

### SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling  <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a> 


</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/open-vocabulary-semantic-segmentation-on-15&color=3F51B5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-15?p=skysense-a-multi-modal-remote-sensing)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/open-vocabulary-semantic-segmentation-on-fast&color=3F51B5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-fast?p=skysense-a-multi-modal-remote-sensing)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/open-vocabulary-semantic-segmentation-on-16&color=3F51B5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-16?p=skysense-a-multi-modal-remote-sensing)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/open-vocabulary-semantic-segmentation-on-sior&color=3F51B5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-sior?p=skysense-a-multi-modal-remote-sensing)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/open-vocabulary-semantic-segmentation-on-sota&color=3F51B5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-sota?p=skysense-a-multi-modal-remote-sensing)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/visual-question-answering-on-siri-whu&color=3F51B5)](https://paperswithcode.com/sota/visual-question-answering-on-siri-whu?p=skysense-a-multi-modal-remote-sensing)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/visual-question-answering-on-aid-vqa&color=3F51B5)](https://paperswithcode.com/sota/visual-question-answering-on-aid-vqa?p=skysense-a-multi-modal-remote-sensing)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/visual-question-answering-on-rsvqa-hr&color=3F51B5)](https://paperswithcode.com/sota/visual-question-answering-on-rsvqa-hr?p=skysense-a-multi-modal-remote-sensing)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/zero-shot-classification-unified-classes-on-2&color=3F51B5)](https://paperswithcode.com/sota/zero-shot-classification-unified-classes-on-2?p=skysense-a-multi-modal-remote-sensing)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/skysense-a-multi-modal-remote-sensing/zero-shot-classification-unified-classes-on-3&color=3F51B5)](https://paperswithcode.com/sota/zero-shot-classification-unified-classes-on-3?p=skysense-a-multi-modal-remote-sensing)  
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.14033-b31b1b.svg)](https://arxiv.org/abs/2312.14033)  [![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](./LICENSE) -->


<div align="left">

## Introduction✨

This is a model aggregated with CLIP and SAM version of SkySense for remote sensing interpretation described in [SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling](https://arxiv.org/abs/2312.14033). In addition to introducing <strong>a powerful remote sensing vision-language foundation model</strong>, we have also proposed <strong>the first open-vocabulary segmentation dataset</strong> in the remote sensing domain. Each ground truth (contains mask and text) in the dataset has undergone multiple rounds of <strong>annotation and validation by human experts</strong>, enabling the capability to <strong style="color:red">segment anything in open remote sensing scenarios</strong>.
<!-- >Open-world interpretation aims to accurately localize and recognize all objects within images by vision-language models (VLMs). While substantial progress has been made in this task for natural images, the advancements for remote sensing (RS) images still remain limited, primarily due to these two challenges.1) Existing RS semantic categories are limited, particularly for pixel-level interpretation datasets.2) Distinguishing among diverse RS spatial regions solely by language space is challenging due to the dense and intricate spatial distribution in open-world RS imagery. To address the first issue, we develop a fine-grained RS interpretation dataset, Sky-SA, which contains 183,375 high-quality local image-text pairs with full-pixel manual annotations, covering 1,763 category labels, exhibiting richer semantics and higher density than previous datasets.Afterwards, to solve the second issue, we introduce the vision-centric principle for vision-language modeling. Specifically, in the pre-training stage, the visual self-supervised paradigm is incorporated into image-text alignment, reducing the degradation of general visual representation capabilities of existing paradigms. Then, we construct a visual-relevance knowledge graph across open-category texts and further develop a novel vision-centric image-text contrastive loss for fine-tuning with text prompts.This new model, denoted as SkySense-O, demonstrates impressive zero-shot capabilities on a thorough evaluation encompassing 14 datasets over 4 tasks, from recognizing to reasoning and classification to localization. Specifically, it outperforms the latest models such as SegEarth-OV, GeoRSCLIP, and VHM by a large margin, i.e., 11.95\%, 8.04\% and 3.55\% on average respectively. -->

<!-- <div>
    <strong style="color: red; font-size: 18px;"> "CLIP and SAM Version of SkySense for Remote Sensing Anything" </strong>
</div> -->

</div>


<!-- <div>
    <h4 align="center">
        • <a href="https://likyoo.github.io/SegEarth-OV/" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a> • <a href="https://colab.research.google.com/drive/1a-NNz_2maesvszk4Xff5PKY02_moPqt6#scrollTo=Pz9QGEcFBGtK" target='_blank'>[Colab]</a> •
    </h4>
</div> -->

<img src="docs/figures/teaser.png" width="100%"/>
<!-- Visualization and performance of SegEarth-OV on open-vocabulary semantic segmentation of remote sensing images. We evaluate on 17 remote sensing datasets (including semantic segmentation, building extraction, road extraction, and flood detection tasks), and our SegEarth-OV consistently generates high-quality segmentation masks. -->

## News 🚀
- `2025/02/27`: 🔥 SkySense-O has been accepted to <strong>CVPR2025</strong> !
- `2025/04/08`: 🔥 We introduce <strong>SkySense-O</strong>, demonstrating impressive zero-shot capabilities on a thorough evaluation encompassing 14 datasets, from recognizing to reasoning and classification to localization. Specifically, it outperforms the latest models such as SegEarth-OV, GeoRSCLIP, and VHM by a large margin, i.e., <strong>11.95\%, 8.04\% and 3.55\%</strong> on average respectively.

## TODO 📝
- [ ] Release the checkpoints, inference codes and demo.
- [ ] Release the dataset and training scripts.
- [ ] Release the evaluation code.
- [ ] Release the code for data generation pipeline.


## Dependencies and Installation


##### 1. install detectron2
```
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
```
##### 2. clone this repository and install dependencies
```
git clone https://github.com/zqcraft/SkySense-O.git
cd SkySense-O
pip install -r require.txt
pip install accelerate -U
```
##### 3. Sky-SA Dataset preparation
<!-- We include the following dataset configurations in this repo: 
1) `Semantic Segmentation`: OpenEarthMap, LoveDA, iSAID, Potsdam, Vaihingen, UAVid<sup>img</sup>, UDD5, VDD
2) `Building Extraction`: WHU<sup>Aerial</sup>, WHU<sup>Sat.Ⅱ</sup>, Inria, xBD<sup>pre</sup>
3) `Road Extraction`: CHN6-CUG, DeepGlobe, Massachusetts, SpaceNet
4) `Water Extraction`: WBS-SI -->
Please refer to [data_engine.md](data_engine.md) for details of Sky-SA dataset.

## Quick Inference
```
sh demo.sh
```
## Model Training
```
sh run_train.sh
```
## Model Evaluation
```
sh run_eval.sh
```



## Results

<div align="center">
<img src="docs/figures/exer.png" width="95%"/>
</div>

## Citation

```
@article{zhu2024skysenseo,
  title={SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling},
  author={Qi Zhu, Jiangwei Lao, Deyi Ji, Junwei Luo, Kang Wu, Yingying Zhang, Lixiang Ru, Jian Wang, Jingdong Chen, Ming Yang, Dong Liu, Feng Zhao},
  journal={arXiv preprint },
  year={2025}
}
```

## Acknowledgement
This implementation is based on [Detectron 2](https://github.com/facebookresearch/detectron2). Thanks for the awesome work.
