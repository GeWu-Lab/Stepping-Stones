# 💬Stepping-Stones
Here is the official PyTorch implementation of ''*Stepping Stones: A Progressive Training Strategy for Audio-Visual Semantic Segmentation*''. Please refer to our [ECCV 2024 paper](https://arxiv.org/abs/2407.11820) for more details.

**Paper Title: "Stepping Stones: A Progressive Training Strategy for Audio-Visual Semantic Segmentation"**

**Authors: [Juncheng Ma](https://ucasmjc.github.io/), Peiwen Sun, Yaoting Wang and [Di Hu](https://dtaoo.github.io/index.html)**

**Accepted by: European Conference on Computer Vision(ECCV 2024)**

🚀: Project page here: [Project Page](https://gewu-lab.github.io/stepping_stones/)

📄: Paper here: [Paper](https://arxiv.org/abs/2407.11820)

🔍: Supplementary material: [Supplementary](https://gewu-lab.github.io/stepping_stones/static/pdfs/09290-supp.pdf)
## Overview
Audio-Visual Segmentation (AVS) aims to achieve pixel-level localization of sound sources in videos, while Audio-Visual Semantic Segmentation (AVSS), as an extension of AVS, further pursues semantic understanding of audio-visual scenes. However, since the AVSS task requires the establishment of audio-visual correspondence and semantic understanding simultaneously, we observe that previous methods have struggled to handle this mashup of objectives in end-to-end training, resulting in insufficient learning and sub-optimization. Therefore, we propose a two-stage training strategy called Stepping Stones, which decomposes the AVSS task into two simple subtasks from localization to semantic understanding, which are fully optimized in each stage to achieve step-by-step global optimization. This training strategy has also proved its generalization and effectiveness on existing methods. To further improve the performance of AVS tasks, we propose a novel framework Adaptive Audio Visual Segmentation, in which we incorporate an adaptive audio query generator and integrate masked attention into the transformer decoder, facilitating the adaptive fusion of visual and audio features.  Extensive experiments demonstrate that our methods achieve state-of-the-art results on all three AVS benchmarks. 

<img width="1009" alt="image" src="image/teaser.png">


## Results
### Quantitative comparision
| Method            | S4    |          | MS3   |          | AVSS  |          | Reference |
|-------------------|-------|----------|-------|----------|-------|----------|-----------|
|                   | *mIoU*  | *F-score*  | *mIoU*  | *F-score* | *mIoU* | *F-score*  |           |
| AVSBench          | 78.7 | 87.9    | 54.0 | 64.5     | 29.8 | 35.2     | ECCV'2022 |
| AVSC              | 80.6 | 88.2    | 58.2 | 65.1    | -     | -        | ACM MM'2023 |
| CATR              | 81.4 | 89.6    | 59.0 | 70.0    | 32.8 | 38.5    | ACM MM'2023 |
| DiffusionAVS      | 81.4 | 90.2    | 58.2 | 70.9    | -     | -        | ArXiv'2023 |
| ECMVAE            | 81.7 | 90.1    | 57.8 | 70.8    | -     | -        | CVPR'2023 |
| AuTR              | 80.4 | 89.1    | 56.2  | 67.2     | -     | -        | ArXiv'2023 |
| SAMA-AVS          | 81.5 | 88.6    | 63.1 | 69.1     | -     | -        | WACV'2023 |
| AQFormer          | 81.6 | 89.4    | 61.1 | 72.1    | -     | -        | IJCAI'2023 |
| AVSegFormer       | 82.1 | 89.9    | 58.4 | 69.3    | 36.7 | 42.0    | AAAI'2024 |
| AVSBG             | 81.7 | 90.4     | 55.1 | 66.8     | -     | -        | AAAI'2024 |
| GAVS              | 80.1 | 90.2     | 63.7 | 77.4     | -     | -        | AAAI'2024 |
| MUTR              | 81.5  | 89.8     | 65.0  | 73.0     | -     | -        | AAAI'2024 |
|**AAVS(Ours)**        | **83.2** | **91.3** | **67.3** | **77.6** | **48.5\*** | **53.2\***   | ECCV'2024 |

>  $^*$ indicates that the model uses the Stepping Stones strategy.
### Quantitative comparision

Single Sound Source Segmentation(S4): 
<img width="1009" alt="image" src="image/s4.png">

Multiple Sound Source Segmentation(MS3):
<img width="1009" alt="image" src="image/ms3.png">

Audio-Visual Semantic Segmentation(AVSS):
<img width="1009" alt="image" src="image/v2.png">


## Code instruction

### Data Preparation
Please refer to the link [AVSBenchmark](https://github.com/OpenNLPLab/AVSBench) to download the datasets. You can put the data under `data` folder or rename your own folder. Remember to modify the path in config files. The `data` directory is as bellow:
```
|--data
   |--v2
   |--v1m
   |--v1s
   |--metadata.csv
```

### Pre-trained backbone
We use Mask2Former model with Swin-B pre-trained on ADE20k as backbone, which could be downloaded in this [link](https://drive.google.com/file/d/15wI-2M3Cfovl6oNTvBSQfDYKf5FmqooD/view?usp=drive_link). Don't forget to modify the path in config.py. 

In addition, we changed some metadata of the backbone, and you should replace the config.json and preprocessor_config.json in ".models" folder by ones provided by us (for avs and avss subtasks respectively).

### Download checkpoints
We provides checkpoints for all three subtasks. You can download them from the following links for quick evaluation.

|Subset|mIoU|F-score|Download|
|:---:|:---:|:---:|:---:|
|S4|83.18|91.33|[ckpt](https://drive.google.com/file/d/1Y8GvGjdwixBcDuv2-JY3aXTnNPQ5Zs3l/view?usp=drive_link)|
|MS3|67.30|77.63|[ckpt](https://drive.google.com/file/d/1rV_7ZDS0OtWQ5aLZulrXJjPjgz5gIuS2/view?usp=drive_link)|
|AVSS|48.50|53.20|[ckpt](https://drive.google.com/file/d/1j0bmGJaacWxSlg1kR86E9Rf9IoA0HBhA/view?usp=drive_link)|

### Testing
At first, you should modify paths in config.py.

For S4 and MS3 subtasks, you can run the following code to test.
~~~shell
cd avs
sh test.sh
~~~
For AVSS subtask, you should put predicted masks without semantic from trained AVSS model in the following format firstly, and modify the ''mask_path'' in config.py. Or you can download results used by us in this [link](https://drive.google.com/file/d/1sHvltXub5Ql_ZsVelE1Zt9TpLeb4i3hq/view?usp=drive_link).

```
|--masks
   |--v2
      |--_aldtLqTVYI_1000_11000
         |--0.png
         |--...
   |--v1m
   |--v1s
```
Then, you can run the following code to test.
~~~shell
cd avss
sh test.sh
~~~
### Training
For S4 and MS3 subtasks, you can run the following code to train:  
> Remember to modify the config.
~~~shell
cd avs
sh train_avs.sh
~~~
For stepping_stones for AVSS subtask, you can run the following code to train:
> Remember to put predicted masks without semantic in the right way and modify the config.
~~~shell
cd avss
sh train_avss.sh
~~~
## Citation
If you find this work useful, please consider citing it.

~~~BibTeX
@article{ma2024steppingstones,
          title={Stepping Stones: A Progressive Training Strategy for Audio-Visual Semantic Segmentation},
          author={Ma, Juncheng and Sun, Peiwen and Wang, Yaoting and Hu, Di},
          journal={IEEE European Conference on Computer Vision (ECCV)},
          year={2024},
         }
~~~
