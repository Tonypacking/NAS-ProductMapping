# Semestral work
- [Experiment results](#experiment-results)
- [Requirenments](#requirenments)
- [Usage](#usage)
- [Documentation](#developer-documentation)
- [Project organisation](#project-organisation)
## Requirenments
Requires at least python 3.12.3 and packages in file RP/requirenments.txt 

Other python or package versions weren't tested.

To get all python packages run:
````
 pip install -r requirements.txt
````

## Project Organisation
This project is split into 3 folders and analyze_prediction.py evolution_vs_backprop.py script and this README.md file.

1. **vrba_adam/RP/Examples**: This folder contains example arguments by which we run this project. 
2. **vrba_adam/RP/ExperimentResults**: This folder contains experiment reuslts
3. **vrba_adam/RP/WeightSearch_Backpropagation**: 
4. **vrba_adam/RP/WeightSearch_Evolution**: 
5. **vrba_adam/Data** This folder contains all datasets required for product mapping/
## Experiment results

To run a sample experiment and generate comprehensive results, follow these steps:

**1. Execute the Experiment:**
Run the following command in your terminal:

```
python3 evolution_vs_backprop.py --o ExampleRun --iter 15 --run_all all --gen 15 --sb ExampleRun/Backpropagation --se ExampleRun/Evolution
```

**2. Understanding the Output:**
Upon successful execution, a new directory named `ExampleRun` will be created in your current working directory. This folder will contain all the experiment's best output, organized as follows:

* **`ExampleRun/Backpropagation` Folder:**
    This subdirectory will contain results and graphs visualizing the accuracy achieved by the backpropagation method for each unique combination of dataset and preprocessing technique.

* **`ExampleRun/Evolution` Folder:**
    This subdirectory will display graphs illustrating the evolution of fitness over generations for the evolutionary approach. 

Backpropagation and Evolution folders will have following subdirectories with applies preprocessing methods.

* **Dataset and Preprocessing Variations:** The experiment is designed to test various training datasets, applying three distinct preprocessing techniques:
    * Linear Discriminant Analysis (LDA)
    * Principal Component Analysis (PCA)
    * No preprocessing (raw data)

This setup allows for a clear comparison of performance across different methodologies and data preparation strategies.

**3. Result comparison**
Evolutionary algorithms can achieve test accuracies comparable to backpropagation methods. However, for larger networks, evolutionary algorithms are significantly slower than gradient-based approaches.
# Experiment configuration
We can edit 
### Arguments
This project supports multiple arguments which edits program

- #### Seed
### Example:
````
python vrba_adam/RP/evolution_vs_backprop.py --seed 42
````
##### Description: Sets a seed to random number generation.

- #### hidden_layers
### Example:
````
python vrba_adam/RP/evolution_vs_backprop.py --hidden_layers "12 12, 10 10, 12"
````
##### Description:  Allows you to define the hidden layer architectures to be searched during the execution. The script will evaluate each configuration and select the best-performing one. This example will setup 3 NN where the first one has two hidden layers of neuron sizes(12,12) , the second has two hidden layers of neuron sizes (10,10) and the last one has just one hidden layer 12

- #### run_all
### Example:
````
python vrba_adam/RP/evolution_vs_backprop.py --run_all all

````
##### Description: Allows 4 possible values
- None (nothing was selected) -> runs the experiment with selected dataset, dimension reduction etc.
- dim -> Runs the experiment with all possible dimension reduction selected
- dataset -> Runs the experiment on all possible ProMap datasets
- all -> Runs the experiment on all possible dimension reduction and datasets.


- #### save_evo
### Example:
````
python vrba_adam/RP/evolution_vs_backprop.py --save_evo Example

````
##### Description: OutputDirectory for evolutionary search results

- #### generations
### Example:
````
python vrba_adam/RP/evolution_vs_backprop.py --generations

````
##### Description: number of generations in evolution search

- #### metrics
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py -- metrics

````
##### Description: A metric which is used to determine fitness of an individual. Allows 7 possible values.
- f1_binary
- f1_weighted
- f1_macro
- f1_micro
- accuracy
- precision
- recall


- #### Iterations
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py --iterations 50

````
##### Description: Number of iterations in gradient_search

- #### save_back
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py --save_back

````
##### Description: OutputDirectory for gradient search results


- #### dimension reduction
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py --dims raw

````
##### Description: Dimension reduction method
- raw -> No dimension reduction is applied.
- lda -> LDA method is used to reduce input dimensions.
- pca -> PCA method is used to reduce input dimensions.

- #### Select dataset
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py -- dataset promapcz

````
##### Description: Select a dataset on which the experiment is run. 

- google -> selects amazon-google dataset
- walmart -> selects amazon-walmart dataset
- promapcz -> selects promapcz dataset
- promapen -> selects promapen dataset
- promapenext -> selects extended promapext dataset
- promapczext -> selects extended promapcz dataset
- amazonext -> selects promapmulti_amazon_ext dataset

- #### output 
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py --output OutputDirectory

````
##### Description:
 Output directory where test set accuracy of all models is saved (It is saved in validation_predictions.csv file).


- #### analyze
### Example:
````
python $HOME/vrba_adam/RP/evolution_vs_backprop.py --analyze

````
##### Description:
If set searching for the best model is disabled and just reads the validation_predictions.csv

# Developer documentation
This project scripts are structured in 4 different folders
- [main classes for the experiment](#vrba_adamrp)
- [Weight search](#vrba_adamrpweightsearch_evolution)
- [Weight search via backpropagation](#vrba_adamrpweightsearch_backpropagation)
-[Utilites](#vrba_adamutils)

## vrba_adam/RP
Contains two files
- [helper class for analyzing predictions](#analyze_predictionspy)

- [main class for this experiment](#evolution_vs_backproppy)

### analyze_predictions.py
Analyze_predictions.py contains one class 
- Analyze


#### Analyze
- **Type:** class

- **summary:** Helper class which analyzes the experiments predictions.
Reads the validation_prediction.csv located at path which was specified from output argument. There it created for each dataset .txt file with information which dataset was used to train the model which dim. reduction was used and the tested datasets and metric's results.

- #### analyze
- - **Type:** function
- **summary:** Analyzes validation predictions file and saves the output at the output path.

### evolution_vs_backprop.py
Contains multiple functions
- log_statistics
- run_all
- main
- parse_tuple_list
#### log_statistics
- **Type:** function
- **summary:** Helper function to log statistics from the model's performance.

#### run_all
- **Type:** function
- **summary:** Helper function to run the experiment on every dataset and dimension reduction

#### main
- **Type:** function
- **summary:** main function for the experiment. Contains main loop where the weight search happens.

#### parse_tuple_list
- **Type:** function
- **summary:** Helper function to parse hidden_layers from arguments.

## vrba_adam/RP/WeightSearch_Backpropagation

#### **Backprop_Weight_Search**
- **Type:** class

- **summary:** This class contains main loop in which weight search via backpropagation happens.

- #### run
- - **Type:** function
- **summary:** This function runs grid search. Identify the best-performing network based on the user's specified arguments.

- #### plot_bestmodel_accuracy_progress
- - **Type:** function
- **summary:** Retrains the best found model and plots the a graph of precision recall f1 score and accuracy.

- #### save_network
- - **Type:** function
- **summary:** Saves the best network

- #### load_model
- - **Type:** function
- **summary:** Loads the best network


- #### validate
- - **Type:** function
- **summary:** Validates NN againsts f1 score (binary, macro, micro and weighted average), precision, recall, accuracy and confusion matrix

- #### validate_all
- - **Type:** function
- **summary:** Runs validate function on every possible dataset. If feature count of trained data and tested data aren't equal it either fills missing features with zeros or remove columns which aren't presents in the training dataset.

## vrba_adam/RP/WeightSearch_Evolution

#### **EvolutionaryNeuronNetwork**
- **Type:** class

- **summary:** Wrapper of sklearn's NN. Enables to change weights of a neuron network and get parameters of the NN.

- #### Get_Metrics
- - **Type:** function
- **summary:** Gets all possible callable metrics which can be used as a fitness function in evolution algorithms

- #### _choose_metric

- - **Type:** function
- **summary:** Chooses specific metric which will be used as a fitness metric in evolution algorithm.

- #### _parameter_count
- - **Type:** function
- **summary:** Counts parameters in neural network

- #### change_weights
- - **Type:** function
- **summary:** Changed weights of a NN.

- #### network_accuracy
- - **Type:** function
- **summary:** Returns accuracy of the network on current weights.

- #### validate
- - **Type:** function
- **summary:** Validates NN againsts f1 score (binary, macro, micro and weighted average), precision, recall, accuracy and confusion matrix

- #### validate_all
- - **Type:** function
- **summary:** Runs validate function on every possible dataset. If feature count of trained data and tested data aren't equal it either fills missing features with zeros or remove columns which aren't presents in the training dataset.

- #### save_network
- - **Type:** function
- **summary:** Saves the best network

- #### load_model
- - **Type:** function
- **summary:** Loads the best network



#### **Evo_WeightSearch**
- **Type:** class

- **summary:** Main algorithm for searching weights in NN via evolution algorithms.

- #### _fitness
- - **Type:** static function
- **summary:** FItness function of evolutionary algorithm.



- #### run
- - **Type:**  function
- **summary:** Runs weight search neuroevolution generation. Uses CMA-ES algorithm for the evolutionary part.

- #### validate_all
- - **Type:**  function
- **summary:** Validates the best network on every possible dataset. If feature count of trained data and tested data aren't equal it either fills missing features with zeros or remove columns which aren't presents in the training dataset.



### vrba_adam/Utils


#### Dataset
- **Type:** class

- **summary:**      Dataset wrapper. 
    Splits training and testing data to features and targets

- #### scale_features
- - **Type:** function
- **summary:** Scales features with  StandartScaler

- #### reduce_dimensions
- - **Type:** function
- **summary:** Reduce dimensions of test and train data
        LDA will compute maximum number of components based on the number of classes in the dataset.
        PCA will select the number of components such that the amount of variance that needs to be explained is greater than 95%

#### ProductsDatasets
- **Type:** class

- **summary:**      Class which makes easier loading and manipulating ProMap datasets.

- #### Load_by_name
- - **Type:** function
- **summary:** Loads dataset by the given name.

- #### __Split_data
- - **Type:** function
- **summary:** Splits data into train and test size. If the training dataset has more features fills the missing features with 0 if the training dataset is missing some features which are present in testing dataset such features are removed from testing dataset.

- #### Load_basic_amazon_google
- - **Type:** function
- **summary:** Loads amazon_google dataset.

- #### Load_basic_amazon_walmart
- - **Type:** function
- **summary:** Loads amazon walmart dataset.

- #### Load_basic_promap_cz
- - **Type:** function
- **summary:** Loads promapcz dataset.

- #### Load_basic_promap_en
- - **Type:** function
- **summary:** Loads promapen dataset.

- #### Load_extended_promap_cz
- - **Type:** function
- **summary:** Loads extended promapcz dataset.

- #### Load_extended_promap_en
- - **Type:** function
- **summary:** Loads extended promapen dataset.

- #### Load_extended_amazon_walmart
- - **Type:** function
- **summary:** Loads extended amazon walmart  dataset.
