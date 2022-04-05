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

-images: \\figure\Analysis.png

We extracted 500 images and counted vehicles,pedestrians,and bicycles.
From the left of the graph,vehicles,pedestrians and bicycles.
The majority of objects in the dataset are vehicles and pedestrians, while fewer bicycles are detected.

![r_Image1](figure\download1.png)
![r_Image2](figure\download2.png)
![r_Image3](figure\download3.png)
![r_Image4](figure\download4.png)
![r_Image5](figure\download5.png)
![r_Image6](figure\download6.png)
![r_Image7](figure\download7.png)
![r_Image8](figure\download8.png)
![r_Image9](figure\download9.png)
![r_Image10](figure\download10.png)

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
#### Reference experiment
Each model is in this folder "\\experiments\experiments*\reference\exported".    

Each gif animation is the file "\\experiments\experiments*\reference\animation.gif".  


#### reference

experiments0
- folder: \report\experiments0
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](\report\experiments0\pipeline_new.config)    

Result
![r_Loss](\report\experiments0\tensorboad\Loss.PNG)
![r_Precision](\report\experiments0\tensorboard\DetectionBoxes_Precision.PNG)
![r_Recall](\report\experiments0\tensorboard\DetectionBoxes_Recall.PNG)

The Loss graphs are gradually decreasing as learning progresses.  
In the case that the model have learned well, the loss of validation is expected to be lower than the loss of training.
And, the precision and recall are expected be large value than shown in this above graph and the below metrics.  


I compared this reference model and the following experiment1/2/3 model. The result is written the below.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

#### experiment1
- folder: \report\experiments1
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](\report\experiments1\pipeline_new.config)    

  I add 3 data_augmentation_options on the reference pipeline.  
  1. random_adjust_brightness
  2. random_adjust_contrast
  3. random_distort_color


Result
![r_Loss](\report\experiments1\tensorboad\Loss.PNG)
![r_Precision](\report\experiments1\tensorboard\DetectionBoxes_Precision.PNG)
![r_Recall](\report\experiments1\tensorboard\DetectionBoxes_Recall.PNG)

In this learning, the loss has hardly decreased and has converged to a very high value.
Therefore learning is not good. 


#### experiment2
- folder: \report\experiments2
- base model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](\report\experiments2\pipeline_new.config)    

  I change the following steps of training_config.  
  total_steps: 5000  
  num_steps: 5000

Result
![r_Loss](\report\experiments2\tensorboad\Loss.PNG)

This training output is too large.  
I have reached the storage limit of 3GB.  



#### experiment3
- folder: \report\experiments3
- base model: ssd_resnet152_v1_fpn_640x640_coco17_tpu-8  
- pipeline: [pipeline_new.config](\report\experiments3\pipeline_new.config)    

I changed the base model.

Result
![r_Loss](\report\experiments3\tensorboad\Loss.PNG)
![r_Precision](\report\experiments3\tensorboard\DetectionBoxes_Precision.PNG)
![r_Recall](\report\experiments3\tensorboard\DetectionBoxes_Recall.PNG)

The Loss is not lower than reference.Therefore,learning is not as good as reference.



