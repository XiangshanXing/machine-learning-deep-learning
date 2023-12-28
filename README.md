for study and explore further application/主要对当前深度算法同步归类，使用和介绍

主要针对当前研究的图像算法，包括识别追踪等归类，学习，以及在使用中总结。深度学习当前应用在工业领域还是局限性很明显，样本受限，准确度无法遇见，而且对于样本不均衡很难处理，迭代何时可以满足需要，难！


方向

1. 追踪识别（YOLOV5\V8和faster RCNN系列(mmdet,detectron2)等）

2. 机器翻译RNN，tranformer等

3. 分割Unet等(主要针对医疗图像Unet resUnet DenseUnet，attentionUnet,实践过程中发现DenseUnet的效果比较好，loss设定是TverskyLoss比较好一点，但是整体效果都差不多，不会有太多实质上的提升，由于层数限制其泛化能力，个人见解：如果简单场景可以，对于复杂场景落地有点难，这个网络只要数据没问题，随便训练都问题不大，这一点很厉害)

4. 交互式网络SAM等


推荐

1. 动手学深度学习：李沐（https://www.bilibili.com/video/BV18h411r7Z7/?spm_id_from=333.999.0.0&vd_source=84771d2107b1399b41545fa9a3f5d2ea）（大神无私，有着家国情怀：达则兼济天下）

2. https://paperswithcode.com/ （和gihub开发必备，常常追踪最新动向）

3. pytorch官网和基于开发的各种应用包
