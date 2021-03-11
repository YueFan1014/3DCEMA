# 3DCEMA
To train a model based on a dataset, please run 'training.py'. The parameters are listed below.
--data_directory: The name of the folder containing 'expressionData.csv' and 'refNetwork.csv'.
--model_name: The name of the model that will be created in the directory 'trained_models\'.
--max_number: The max limit of the training matrix number. Default is 10000.
--num_of_classes: The number of classes. For in-silico datasets it's 3; for scRNA-seq datasets it's 2. Default is 3.
--learning_rate: The learning rate. Default is 0.01.

Here is an example:

training.py 
--data_directory 	scRNA-seq-datasets\mESC\second_100\ 
--model_name 	mESC100t 
--max_limit 	3000 
--num_of_classes 	2

This command will automatically create 3000 training matrices in directory  'scRNA-seq-datasets\mESC\second_100\dataset\'. It will create a trained model named 'mESC100t' in directory 'trained models\'.


To infer a GRN based on a dataset, please run 'inferring.py'. The parameters are listed below.
--data_directory: The folder name containing 'expressionData.csv' and 'refNetwork.csv'.
--model_name: The name of the model that has been created in the directory 'trained_models\'.
--num_of_classes: The number of classes. For in-silico datasets it's 3; for scRNA-seq datasets it's 2. Default is 3.

Here is an example:

infering.py 
--data_directory 	scRNA-seq-datasets\mESC\100\ 
--model_name 	mESC100t 
--num_of_classes 	2

This command will use the trained model named 'mESC100t' to infer the GRN of the dataset named 'mESC100'. A file named 'rankedEdges.csv' will be produced in directory 'scRNA-seq-datasets\mESC\100\'.
