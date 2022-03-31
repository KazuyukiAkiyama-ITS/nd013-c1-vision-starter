## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. 
Why is object detection such an important component of self driving car systems?

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. 
It should include images, charts and other visualizations.


#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. 
It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. 
It should contain relevant figures and details of your findings.




### Project overview
The repository contains the files of the self-driving car project by Udacity. 
The main objective of this project is to detect the objects like cars, pedestrians, and cyclists.
I have used the project workspace provided by Udacity to perform the detection and data exploration. 
The Waymo dataset is pre-loaded into the workspace as tfrecord files, and it is analyzed and then split into train, 
validation and test files. The config file is created using a pre-trained SSD Resnet 50 640x640 model by python programming.
The config file was used to train the waymo data and validate it.

### Set up
Since I have used the Udacity workspace for execution of program, I haven't used setup. 


### Dataset
The dataset is in tfrecord format, and the waymo vehicle recorded the data. 
The data captured were in different weather, different lighting condition, different location. 
The images vary from blur to clear and bright to dark. The rectangle bounding boxes are drawn over on the detected objects in images.

#### Dataset analysis
The below bar chart represents the distribution of classes over 500 images.
The number of car classes from the label is greater than pedestrians and cyclists. 
The cyclists class is fewer when compared to pedestrians, and in the images, the cars and pedestrians classes dominate.

Analysis.png

The captured images are from daytime busy city streets, expressways, uphill roads, and foggy weather. 
The cars are drawn with red anchor boxes while pedestrians in blue and cyclists in green. 

download1.png
download2.png
download3.png
download4.png
download5.png
download6.png
download7.png
download8.png
download9.png
download10.png


The majority of objects in the dataset are cars and pedestrians, while fewer cyclists are detected.

#### Cross validation
The datasets are randomly shuffled so that the images won't get trained for a similar pattern, and images are equally shuffled.
70% of the datasets were split for training, 20% for validation, and 10% for for testing. 


### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.
The trained data plot can be seen in tensorboard folder.

