This is a project in the framework of the MSc course *Social Network Analysis for Computer Scientists*, of Leiden University.

### Abstract

In a group or a community where people interact and form associations with each other, it usually holds that the number of people that are connected is dramatically less than those that do not. Representing such a community as a social network, the aim is to predict those pairs of nodes that will connect in the future. This is known as the *Link Prediction* problem. Utilizing supervised learning, the latter can be phrased as binary classification. In this scheme, pairs of nodes that are connected represent the positive class, whereas those that do not correspond to the negative. Hence, there is a great imbalance between the two classes since the number of links that do not exist is overwhelmingly greater than those that do. Classification with imbalanced data can pose a problem because many algorithms tend to classify every observation in the majority class. In this paper, we apply Hellinger Distance Decision Trees (HDDT), which are considered to be skew insensitive and well suited for imbalanced data, to seven data sets constructed from five social networks. We compare HDDT to C4.5 which is the standard choice of decision trees in the machine learn-
ing community. We evaluate our results using the Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curve.

### Data

The paper can be found in the `report.pdf` file. We provide the `python` code and all the data sets that have been used in this analysis, for anyone that would like to reproduce/improve/study the results.

The data sets can be found as text files in the `datasets` folder. Each data set has been processed using the NetworkName_calculate_metrics.py script in the `code` folder. For each data set, eight metrics are calculated which are then used as features in a HDDT and C4.5 tree. Specifically, the following metrics are calculated using python's `linkpred` library:    
* *Adamic Adar*     
* *Common Neighbors*    
* *Rooted Page Rank*     
* *Jaccard Similariy*    
* *NMeasure*     
* *Minimum Overlap*      
* *Resource Allocation*     
* *Assocation Strength*.     

After calculation, each of these metrics can be loaded using the NetworkName_load_metrics.py script. The HDDT and C4.5 trees are implemented in WEKA. Due to the huge imbalance in some of these data sets, we have under sampled the data to an imbalance of 1:10, and saved them to .arff files. These files can be loaded immediately to WEKA in order to reproduce the results found in the paper. A version of WEKA that contains an implementation of HDDT can be found [https://www3.nd.edu/~dial/software/] (here). The WEKA configuration files can be found in the `weka_config_files` folder.
  
