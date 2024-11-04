# Deep Learning for image analysis


In this seminar the students will learn about recent methods in the field of deep learning for image analysis with a focus on biomedical images. Each week the students will read and understand 2 seminal research papers of a specific subtopic and present and discuss them within the group. 

The seminar will cover a range of topics including image classification, object detection and segmentation, object tracking, and generative modeling. Additionally, we will cover fundamental DL model architectures such as CNNs, Transformers, and Diffusion models.

Course language will be english.

General Structure: 

- 3-4 papers per session/topic covering  
- Intro (10 mins)
- Presentation F+A (2x20 mins) 
- Discussion and Q&A (30 mins)

Date: Friday 4 DS
Place: [N63/A001]([url](https://navigator.tu-dresden.de/etplan/n63/00/raum/191100.0020))
OPAL: https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/46688141312?19


# Session1 - Basic CNNs


This session will cover the basic concepts of Convolutional Neural Networks (CNNs), exploring their architecture and the essential components that make them effective for image analysis. Topics will include different fundamental  architectures such as U-Nets or ConvNeXt. By understanding these basics, students will gain insights into how CNNs extract spatial features from images.

### Key papers:

- He et al. “Deep Residual Learning for Image Recognition.” CVPR (2016)
- Falk et al. "U-Net: deep learning for cell counting, detection, and morphometry" Nat Methods  (2019). 
- Liu et al. "A ConvNet for the 2020s" CVPR (2022)
- Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions" CVPR (2023)


<!-- - Convolutional Neural Networks -> basics  
- Architecture. Convolutions, Pooling, Activation functions, Deformable Convolutions 
- Papers: AlexNet, Resnets, UNets, activations    -->


# Session2 - Low Level Vision 

In this session, we will look into low-level vision tasks, focusing on image denoising and deblurring. These tasks are common for microscopy images that often are recorded at low light conditions and are resolution limited by the optics. This session will introduce techniques and models aimed at these goals.


### Key papers:

- Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising." IEEE TIP  (2017)
- Krull et al. "Noise2void-learning denoising from single noisy images." CVPR (2019)
- Chen et al. "Simple baselines for image restoration." ECCV (2022)


# Session3 - Vision Transformer

This session introduces Vision Transformers (ViTs), which apply the transformer architecture to visual data. We will cover the basics of attention mechanisms, positional embeddings, and the unique attributes that make ViTs suitable for image analysis. 

### Key papers:

- Vaswani et al. “Attention is All You Need.” NeurIPS (2017)
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR (2021)
- Su et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing (2024)



# Session4 - Semantic Segmentation 


In this session, we explore semantic segmentation, where each pixel in an image is classified into a specific category. This is an importnant subtask  in medical and bioimage analysis as well as for pose estimation. We will discuss popular models like  DeepLab, and HRNet, as well as approaches based on ViTs.


### Key papers:

- Chen et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." IEEE TPAMI (2017)
- Sun et al. "Deep high-resolution representation learning for human pose estimation." CVPR (2019)
- Cheng et al. "Per-pixel classification is not all you need for semantic segmentation." NeurIPS (2021)


# Session5 - Detection and Instance Segmentation 

This session will cover object detection and instance segmentation. We will examine models like Mask R-CNN and YOLO variants, which are among the most popular architectures for this task. The Segment Anything model will also be introduced, a transformer based foundation model for this task 


### Key Papers:

- He et al. "Mask R-CNN." CVPR (2017)
- Bochkovskiy et al. "Yolov4: Optimal speed and accuracy of object detection." arXiv (2020)
- Kirillov et al. "Segment anything." CVPR (2023)

# Session6 - Object Tracking 

In this session, we focus on object tracking, an important task especially for bioimage analysis. Object tracking involves following an object's movement across video frames. We will discuss Siamese-based and transformer-based tracking models, as well as the second iteration of the Segment Anything model.

### Key Papers:

- Zhang et al. "Bytetrack: Multi-object tracking by associating every detection box." ECCV (2022)
- Meinhardt et al. "Trackformer: Multi-object tracking with transformers." CVPR (2022)
- Ravi et al. "Sam 2: Segment anything in images and videos." arXiv (2024)

# Session7 - Self Supervised Learning

Self-supervised learning has become a popular approach in computer vision, particularly when labeled data is limited. In this session, we cover the basics of self-supervised learning, with a focus on contrastive learning techniques. Models like MAE, SimCLR, and Dino v2 will be discussed, showcasing the potential of self-supervision in representation learning.

### Key Papers:

- Chen et al. "A simple framework for contrastive learning of visual representations." ICML (2020)
- He  et al. "Masked autoencoders are scalable vision learners."  CVPR (2022)
- Oquab et al. "Dinov2: Learning robust visual features without supervision." Transactions on Machine Learning Research (2023)


# Session8 - Generative Models 1

This session introduces generative models, starting with Generative Adversarial Networks (GANs) and their variants. We will explore models such as conditional GANs, CycleGAN, and StyleGAN, which are widely used for image generation and style transfer.


### Key Papers:

- Goodfellow et al. "Generative Adversarial Networks" arXiv (2014)
- Mirza et al "Conditional generative adversarial nets." arXiv  (2014)
- Zhu et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." CVPR (2017)


# Session9 - Generative Models 2

In this session, we continue with generative models, focusing on more advanced techniques such as latent diffusion and score matching. These methods have shown remarkable capabilities in high-quality image generation, pushing the boundaries of what generative models can achieve.

### Key papers:

- Song et al. "Generative modeling by estimating gradients of the data distribution." NeurIPS (2019)
- Song et al "Score-Based Generative Modeling through Stochastic Differential Equations" ICLR (2021)
- Rombach et al. "High-resolution image synthesis with latent diffusion models." CVPR (2022)

# Session10 - Multimodality

The session covers multimodality, focusing on models that handle multiple data types, such as images and text. We will introduce CLIP, SigLIP, which have been groundbreaking in aligning visual and textual data, enabling applications in cross-modal retrieval and zero-shot classification. 

### Key papers:

- Radford et al. "Learning transferable visual models from natural language supervision." ICML (2021)
- Zhai et al. "Sigmoid loss for language image pre-training." CVPR (2023)
- Deitke et al. "Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models." arXiv (2024).




