#3DCEMA
This is the code of paper **Gene Regulatory Network Inference using
3D Convolutional Neural Network**. 3D Co-Expression Matrix Analysis
(3DCEMA) predicts regulatory relationships by classifying 3D co-expression matrices of gene triplets using a 3D
convolutional neural network.

##Training
To train a model based on a dataset, please run 'training.py'. The parameters are listed below.

--data\_directory: The name of the folder containing 'expressionData.csv' and 'refNetwork.csv'.<br/>
--model\_name: The name of the model that will be created in the directory 'trained_models\'.<br/>
--max_number: The max limit of the training matrix number. Default is 10000.<br/>
--num\_of\_classes: The number of classes. For in-silico datasets it's 3; for scRNA-seq datasets it's 2. Default is 3.
--learning_rate: The learning rate. Default is 0.01.

Here is an **example**:

training.py 
<br>--data\_directory 	scRNA-seq-datasets\mESC\second_100\ 
<br>--model\_name 	mESC100t 
<br>--max\_limit 	3000 
<br>--num\_of\_classes 	2

This command will automatically create 3000 training matrices in directory  'scRNA-seq-datasets\mESC\second_100\dataset\'. It will create a trained model named 'mESC100t' in the directory 'trained models\'.

##Inferring
To infer a GRN based on a dataset, please run 'inferring.py'. The parameters are listed below.

--data\_directory: The folder name containing 'expressionData.csv' and 'refNetwork.csv'.<br/>
--model\_name: The name of the model that has been created in the directory 'trained_models\'.<br/>
--num\_of\_classes: The number of classes. For in-silico datasets it's 3; for scRNA-seq datasets it's 2. Default is 3.<br/>

Here is an **example**:

infering.py 
<br>--data\_directory 	scRNA-seq-datasets\mESC\100\ 
<br>--model\_name 	mESC100t 
<br>--num\_of\_classes 	2

This command will use the trained model named 'mESC100t' to infer the GRN of the dataset named 'mESC100'. A file named 'rankedEdges.csv' will be produced in directory 'scRNA-seq-datasets\mESC\100\'.
