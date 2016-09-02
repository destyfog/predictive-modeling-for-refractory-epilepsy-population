# predictive-modeling-for-refractory-epilepsy-population
Defined cohort, extracted features from medical record data and implemented predictive modeling for refractory epilepsy population, to predict the patient’s drug resistance in early stage.

For the pig script, we used in on the AWS platform and the s3://8803bdh is the path storing all the data. To run the script, simply create a cluster on AWS and add the script to the step. The code will automaticly output a training set and a testing set by the random split of ratio of 1:4.
For the python code, simply use py modeling.py to run. In the default setting the event2 data is used and all the matrics of all the models are shown. Also all of the ROC curve we be outputed. Feel free to adjust the main function to run different models.

youtube link: https://www.youtube.com/watch?v=i1HzxmMZeAQ
