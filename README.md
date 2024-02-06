## Zero-shot learning for skeleton-base action recognition using GCN-LSTM model (ZSL-GCN-LSTM)

### Aim and Objectives
This project aims to develop an application featureing human action recognition. Users are allowed to activate assigned commands when they perform specific actions.

There are four objectives in this project:
1. Build a prototype to simulate a real working environment 
    * Stream camera RGB video to PC 
2. Pre-process raw recordings and datasets
    * Body 3D pose reconstruction and estimation using existing models (e.g. OpenPose).
    * Requst [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset license
    * Export 3D skeleton data 
3. Design neural network model and tune hyperparameters
    * Design ZSL-GCN-LSTM neural network architecture
    * Hyperparameter optimization
    * Save model
4. Integrate model and application
    * Embed model into application
    * Integration testing and performance evaluation


### Project Plan
The practical work is planned to start at the mid of February 2024 and is expected to be completed in two to three semesters.
1. Build Prototype \
  A platform dedicated to outputting native camera RGB images and visualizing human body skeleton will be built initially. 45 days are required to build a cross-platform application. The real-time pose detection window is to be added later in this stage.
2. Pose Estimation and export 3D skeleton data \
  This study is based on skeleton-based human action recognition. Two approaches are taken to obtain native kinematic model. One approach is to use an algorithm to convert a 2D image into 3D object by estimating an additional Z-dimension to the prediction. Installing required libraries and configuring existing models for pose estimation and skeleton reconstruction can acheive good performance with less time. The algorithm can be referred to reference paper. The other is to use benchmark datasets that contain 3D human skeletal data directly. [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) is a large-scale dataset specifically designed for human action recognition that contains 3D skeletal data. In this project, hybrid mode is taken. The content of the datasets will first be fed into the neural network model.
3. Design neural network model and tune hyperparameters \
  Due to the strong correlation between the joints of the human body, the structure of the human body skeleton can be regarded as a graph. Joints as vertexes and their natural connections in the human body as edges. In this study, long-short term memory (RNN-based) will be used to learn spatial and temporal features. \
  Zero-shot learning aims to recognize objects whose instances may not have been seen during training. ZSL method can help us solve the problem that similar actions are easily predict wrongly. It is essential to replace action classes to attributes, which requires time and careful consideration.\
  This section is expected to start at the end of the 2023/2024 spring semester and will have a duration of up to 6 months, with the possibility of extension.
4. Application Integration \
  Integrate saved model with application. Enable human action recognition. It is planned to be carried out from the beginning of August to the end of September. The application development time needs to be determined according to the actual situation.
5. Flextime \
  Time for reporting, collecting statistics or continuing doing incomplete sections.

```mermaid
gantt
  title ZSL-GCN-LSTM Plan
  dateFormat YYYY-MM-DD
  section Groundwork
    Topic Analysis                :active, g1, 2024-01-15, 2024-05-28
    Dataset License Request       : g2, after doc1, 30d
  section Project
    Build Prototype               : p1, after doc1, 45d
    Pose estimation               :crit, p2, 2024-02-25, 45d
    Export 3D skeleton data       : p3, after g2, 80d
    Neural Network Design and fine-tune        : p4, 2024-04-15, 2024-10-01
    Application integration       : p5, 2024-08-01, 2024-10-01
  section Documentation
    Registration form submission  :milestone, doc1, 2024-02-09, 0d
    Dissertation Proposal         :crit, active, doc2, after doc1, 2024-03-28
    Progress report               : doc3, 2024-09-01, 40d
    Progress report submission    :milestone, 2024-10-13, 0d
    Dissertation submission       :milestone, 2024-12-20, 0d
  section Presentation
    Presentation 1                :milestone, pre1, 2024-04-25, 0d
    Presentation 2                :milestone, pre2, 2024-05-30, 0d
    Presentation 3                :milestone, pre3, 2024-06-27, 0d
    Presentation 4                :milestone, pre4, 2024-07-23, 0d
    Presentation 5                :milestone, pre5, 2024-09-13, 0d

```

### MindMap
```mermaid
mindmap
  root((ZSL-GCN-LSTM))
    ((HybridNet))
      GFEM
      basic unit
        ((2s-AGCN))
          ((ST-GCN))
            {{Edge Set}}
              [intra-body edges]
              [inter-frame edges]
            {{Partition Strategies}}
              [Uni-labeling <br/>
              < Kipf and Welling 2017 >]
              [Distance partitioning]
              [Spatial configuration partitioning]
                (root = 0)
                (centripetal group = 1)
                (centrifugal group = 2)
            {{Network architecture}}
              (dropout = 0.5)
              (BN layer <br/>
              3 x 64 Channels <br/>
              1 x 128 Channels, stride = 2 <br/>
              2 x 128 Channels <br/>
              1 x 256 Channels, stride = 2 <br/>
              2 x 256 Channels <br/>
              Global Average Pooling <br/>
              FC <br/>
              Softmax)
              (ResNet for each ST-GCN)
              (SGD)
              (init_lr = 0.01 <br/>
              decay_rate = 0.1 for every 10 epochs)
      gluing unit
      CFPM
    ((SMIE))
      modules
        Global alignment module
          Mutual information estimation & maximization
            Statistical correlations
              visual
              semantic distributions
        Temporal constraint module
          Connection model
            Action dynamics exploration
            Inherent temporal information capture
    ((AGC-LSTM))
```

### Reference
[1] H.-C. Nguyen, T.-H. Nguyen, R. Scherer, and V.-H. Le, “Deep Learning for Human Activity Recognition on 3D Human Skeleton: Survey and Comparative Study,” Sensors (Basel), vol. 23, no. 11, p. 5121, May 2023, doi: 10.3390/s23115121. \
[2] C. Si, W. Chen, W. Wang, L. Wang, and T. Tan, “An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA: IEEE, Jun. 2019, pp. 1227–1236. doi: 10.1109/CVPR.2019.00132. \
[3] Y. Xian, C. H. Lampert, B. Schiele, and Z. Akata, “Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 9, pp. 2251–2265, Sep. 2019, doi: 10.1109/TPAMI.2018.2857768. \
[4] C. H. Lampert, H. Nickisch, and S. Harmeling, “Attribute-Based Classification for Zero-Shot Visual Object Categorization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 3, pp. 453–465, Mar. 2014, doi: 10.1109/TPAMI.2013.140. \
[5] T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks.” arXiv, Feb. 22, 2017. Accessed: Jan. 31, 2024. [Online]. Available: http://arxiv.org/abs/1609.02907