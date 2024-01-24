# SSD-based Object Detection in PyTorch

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) in PyTorch for object detection, using MobileNet backbones.  It also has out-of-box support for retraining on Google Open Images dataset and Pascal VOC.  

For documentation, please refer to the following:
* [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md) from the **[Hello AI World](https://github.com/dusty-nv/jetson-inference/tree/master#training)** tutorial <br/>
* [Original Readme](https://github.com/qfgaohao/pytorch-ssd) from [`https://github.com/qfgaohao/pytorch-ssd`](https://github.com/qfgaohao/pytorch-ssd)

> Thanks to @qfgaohao for the [upstream implementation](https://github.com/qfgaohao/pytorch-ssd)


#[캔 모델 실행방법]

- 경로 이동
cd python/training/detection/ssd/


캔 모델은 cafe로 지정해 두었습니다.
-명령어
detectnet --model=models/cafe/ssd-mobilenet.onnx --labels=models/cafe/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes csi://0


