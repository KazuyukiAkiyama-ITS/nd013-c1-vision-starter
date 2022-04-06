# [Submission_report] Object Detection in an Urban Environment

### Project overview
The purpose of this project is that the environment is built and the accuracy and recall of object detection in an urban environment are improved.
The procedure is as the follows.
1. Exploratory Data Analysis: We analysys the datasets and recognize features and trends.
2. Set pipeline: We modify the pipeline config of model, training and eval.
3. Training: We train the model with datasets.
4. Evaluation: We evaluate the model with datasets.
5. Improve model: We repeat steps from 2 to 4 and compare the output
6. Report:we export trained model and create video for model's inferences

### Set up
I'm using the Workspace provided by UDACITY lesson.
Therefore dataset is ready.



Precedure 1: Exploratory Data Analysis

Install Chrome browser
```
sudo apt-get update
sudo apt-get install chromium-browser
sudo chromium-browser --no-sandbox
```
Start Jupyter notebook
```
cd /home/workspace/
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
```
Enter address of Jupyter notebook on Chrome.
Open "Exploratory Data Analysis.ipynb" on Jupyter notebook.

Precedure 2:
1. Split dataset to the group of training, evaluation and test.
The datasets are randomly shuffled so that the images won't get trained for a similar pattern, and images are equally shuffled.
70% of the datasets were split for training, 20% for validation, and 10% for for testing. 
```
python create_splits.py --data-dir /home/workspace/data
```
2. Download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

3. Edit the config file from pretrained model. A new config file `pipeline_new.config` has been created in `home/workspace`.
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
4. Move the file `pipeline_new.config` to `/home/workspace/experiments/reference`

Precedure 3:

Do Training
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

Precedure 4:

Do Evaluation
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference
```
See in TensoBoard
```
python -m tensorboard.main --logdir experiments/reference/
```

Change checkpoints: Change the first line with parameter name as 'model_checkpoint_path' in `/home/workspace/experiments/reference/checkpoint`.

Precedure 5:

Create a new folder named `experiment0`, `experiment1`, and so on in `/home/workspace/experiments`.  
Change the path of command from `reference` to `experiment0`.
Repeat steps from 2 to 4.


Precedure 6:

Export the trained model
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```
Create video
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

### Dataset
#### Dataset analysis

Dataset analysis is written in the following file.  
[Exploratory Data Analysis.ipynb]


![r_Analysis](/figure/Analysis2.png)

We extracted 500 images and counted vehicles,pedestrians,and bicycles.
From the left of the graph,vehicles,pedestrians and bicycles.
The majority of objects in the dataset are vehicles and pedestrians, while fewer bicycles are detected.

![r_Image1](/figure/download1.png)
![r_Image2](/figure/download2.png)
![r_Image3](/figure/download3.png)
![r_Image4](/figure/download4.png)
![r_Image5](/figure/download5.png)
![r_Image6](/figure/download6.png)
![r_Image7](/figure/download7.png)
![r_Image8](/figure/download8.png)
![r_Image9](/figure/download9.png)
![r_Image10](/figure/download10.png)


The captured images are from daytime busy city streets, expressways, uphill roads, and foggy weather. 
The cars are drawn with red anchor boxes while pedestrians in blue and bicycles in green. 


#### Cross validation
The tf records files are splited between these three folders with [`create_splits.py`](create_splits.py).  
from  
`/home/workspace/data/waymo/training_and_validation/`  
to  
`/home/workspace/data/train/`,  
`/home/workspace/data/val/`, and  
`/home/workspace/data/test/`

The detail is in Precedure 2-1.

The split ratio is the following.  
| folder name  | usage      | ratio |
| ----         | ----       | ----  |
| train        | training   | 70%   |
| eval         | validation | 20%   |
| test         | test       | 10%   |

The purpose of cross validation is model generalization and alleviatation the overfitting challenges.  
The test data is different from training and validation data.

### Training
#### reference


experiments0
- folder: /report/experiments0
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](/report/experiments0/pipeline_new.config)    

Result
![r_Loss](/report/experiments0/tensorboard/Loss.png)
![r_Precision](/report/experiments0/tensorboard/DetectionBoxes_Precision.png)
![r_Recall](/report/experiments0/tensorboard/DetectionBoxes_Recall.png)

The Loss graphs are gradually decreasing as learning progresses.  
In the case that the model have learned well, the loss of validation is expected to be lower than the loss of training.
And, the precision and recall are expected be large value than shown in this above graph and the below metrics.  

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.002

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.004

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.029

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.067

Eval metrics at step 2500

DetectionBoxes_Precision/mAP: 0.000508

DetectionBoxes_Precision/mAP@.50IOU: 0.001660

DetectionBoxes_Precision/mAP@.75IOU: 0.000190

DetectionBoxes_Precision/mAP (small): 0.000495

DetectionBoxes_Precision/mAP (medium): 0.000248

DetectionBoxes_Precision/mAP (large): 0.003614

DetectionBoxes_Recall/AR@1: 0.000475

DetectionBoxes_Recall/AR@10: 0.000831

DetectionBoxes_Recall/AR@100: 0.008355

DetectionBoxes_Recall/AR@100 (small): 0.000155

DetectionBoxes_Recall/AR@100 (medium): 0.028660

DetectionBoxes_Recall/AR@100 (large): 0.066616

Loss/localization_loss: 0.871702

Loss/classification_loss: 0.868469

Loss/regularization_loss: 1.851852

Loss/total_loss: 3.592023

I compared this reference model and the following experiment1/2/3 model. The result is written the below.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

#### experiment1
- folder: /report/experiments1
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](/report/experiments1/pipeline_new.config)    

  I add 3 data_augmentation_options on the reference pipeline.  
  1. random_adjust_brightness
  2. random_adjust_contrast
  3. random_distort_color


Result
![r_Loss](/report/experiments1/tensorboard/loss.png)
![r_Precision](/report/experiments1/tensorboard/DetectionBoxes_Precision.png)
![r_Recall](/report/experiments1/tensorboard/DetectionBoxes_Recall.png)

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.004

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.002

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.013

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.008

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.062

Eval metrics at step 2500

DetectionBoxes_Precision/mAP: 0.000232

DetectionBoxes_Precision/mAP@.50IOU: 0.001113

DetectionBoxes_Precision/mAP@.75IOU: 0.000028

DetectionBoxes_Precision/mAP (small): 0.000902

DetectionBoxes_Precision/mAP (medium): 0.003909

DetectionBoxes_Precision/mAP (large): 0.001032

DetectionBoxes_Recall/AR@1: 0.000262

DetectionBoxes_Recall/AR@10: 0.002175

DetectionBoxes_Recall/AR@100: 0.012917

DetectionBoxes_Recall/AR@100 (small): 0.011522

DetectionBoxes_Recall/AR@100 (medium): 0.008003

DetectionBoxes_Recall/AR@100 (large): 0.062043

Loss/localization_loss: 0.887194

Loss/classification_loss: 0.736561

Loss/regularization_loss: 2.946652

Loss/total_loss: 4.570406

In this learning, the loss has hardly decreased and has converged to a very high value.
Therefore learning is not good. 



#### experiment2
- folder: /report/experiments2
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](/report/experiments2/pipeline_new.config)    

  I change the following steps of training_config.  
  total_steps: 5000  
  num_steps: 5000

Result
![r_Loss](/report/experiments2/tensorboard/Loss.png)

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.005

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.116

Eval metrics at step 4000

DetectionBoxes_Precision/mAP: 0.000049

DetectionBoxes_Precision/mAP@.50IOU: 0.000171

DetectionBoxes_Precision/mAP@.75IOU: 0.000018

DetectionBoxes_Precision/mAP (small): 0.000000

DetectionBoxes_Precision/mAP (medium): 0.000085

DetectionBoxes_Precision/mAP (large): 0.000892

DetectionBoxes_Recall/AR@1: 0.000012

DetectionBoxes_Recall/AR@10: 0.000912

DetectionBoxes_Recall/AR@100: 0.004987

DetectionBoxes_Recall/AR@100 (small): 0.000000

DetectionBoxes_Recall/AR@100 (medium): 0.001139

DetectionBoxes_Recall/AR@100 (large): 0.116311

Loss/localization_loss: 0.759206

Loss/classification_loss: 0.943312

Loss/regularization_loss: 2.135895

Loss/total_loss: 3.838413

This training output is too large.  
I have reached the storage limit of 3GB.  



#### experiment3
- folder: /report/experiments3
- base model: ssd_resnet152_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](/report/experiments3/pipeline_new.config)    

I changed the base model.
  total_steps: 1400  
  num_steps: 1400

Since the output was large,the step is reduced.

Result
![r_Loss](/report/experiments3/tensorboard/Loss.png)
![r_Precision](/report/experiments3/tensorboard/DetectionBoxes_Precision.png)
![r_Recall](/report/experiments3/tensorboard/DetectionBoxes_Recall.png)




 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.004

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.091

Eval metrics at step 1000

DetectionBoxes_Precision/mAP: 0.000034

DetectionBoxes_Precision/mAP@.50IOU: 0.000117

DetectionBoxes_Precision/mAP@.75IOU: 0.000007

DetectionBoxes_Precision/mAP (small): 0.000000

DetectionBoxes_Precision/mAP (medium): 0.000000

DetectionBoxes_Precision/mAP (large): 0.000530

DetectionBoxes_Recall/AR@1: 0.000000

DetectionBoxes_Recall/AR@10: 0.000256

DetectionBoxes_Recall/AR@100: 0.003731

DetectionBoxes_Recall/AR@100 (small): 0.000000

DetectionBoxes_Recall/AR@100 (medium): 0.000033

DetectionBoxes_Recall/AR@100 (large): 0.090854

Loss/localization_loss: 0.936573

Loss/classification_loss: 1.078339

Loss/regularization_loss: 6.681696

Loss/total_loss: 8.696609

The Loss is not lower than reference.Therefore,learning is not as good as reference.

