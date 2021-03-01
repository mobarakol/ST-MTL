# ST-MTL Segmentation
An interactive demo is available here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mobarakol/ST-MTL/blob/main/ST_MTL_Segmentation_Demo.ipynb)

This repository contains only segmentation branch of the paper [ST-MTL: Spatio-Temporal multitask learning model to predict scanpath while tracking instruments in robotic surgery](https://www.sciencedirect.com/science/article/pii/S1361841520302012)<br>
Representation learning of the task-oriented attention while tracking instrument holds vast potential in image-guided robotic surgery. Incorporating cognitive ability to automate the camera control enables the surgeon to concentrate more on dealing with surgical instruments. The objective is to reduce the operation time and facilitate the surgery for both surgeons and patients. We propose an end-to-end trainable Spatio-Temporal Multi-Task Learning (ST-MTL) model with a shared encoder and spatio-temporal decoders for the real-time surgical instrument segmentation and task-oriented saliency detection. In the MTL model of shared-parameters, optimizing multiple loss functions into a convergence point is still an open challenge. We tackle the problem with a novel asynchronous spatio-temporal optimization (ASTO) technique by calculating independent gradients for each decoder. We also design a competitive squeeze and excitation unit by casting a skip connection that retains weak features, excites strong features, and performs dynamic spatial and channel-wise feature recalibration. To capture better long term spatio-temporal dependencies, we enhance the long-short term memory (LSTM) module by concatenating high-level encoder features of consecutive frames. We also introduce Sinkhorn regularized loss to enhance task-oriented saliency detection by preserving computational efficiency. We generate the task-aware saliency maps and scanpath of the instruments on the dataset of the MICCAI 2017 robotic instrument segmentation challenge. Compared to the state-of-the-art segmentation and saliency methods, our model outperforms most of the evaluation metrics and produces an outstanding performance in the challenge.<br>

Our contributions are summarized as follows:<br>
– We propose a spatio-temporal MTL (ST-MTL) model with a weight-shared encoder and task-aware spatio-temporal decoders.<br>

–We introduce a novel way to train the proposed MTL model by using asynchronous spatio-temporal optimization (ASTO).<br>

–Our novel design of decoders, skip competitive scSE unit, and ConvLSTM++ boost up the model performance.<br>

–We generate task-oriented instruments saliency and scanpath similar to the surgeon’s visual perception to get the priority focus on surgical instruments. Our model achieves impressive results and surpasses the existing state-of-the-art models in MICCAI robotic instrument segmentation dataset.<br>

For Training:
```
python main.py
```
For Validation:
```
python deploy.py
```

## Citation
If you use this code for your research, please cite our paper.

```
@article{islam2021st,
  title={ST-MTL: Spatio-Temporal multitask learning model to predict scanpath while tracking instruments in robotic surgery},
  author={Islam, Mobarakol and Vibashan, VS and Lim, Chwee Ming and Ren, Hongliang},
  journal={Medical Image Analysis},
  volume={67},
  pages={101837},
  year={2021},
  publisher={Elsevier}
}
```
