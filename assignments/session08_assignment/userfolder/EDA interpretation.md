EDA Interpretations: 

Inspectation: 
there are no missing values
there are no duplicates 

Visualizations: 
Pairplot shows that some features correlate more, while others seem to be more scattered. However, in almost all cases, the wine classes seem to cluster and not be randomly distributed. For example, flavanoids show relatively distinct distributions across wine classes and could be useful for classification
Boxplot shows that wine class 1 seems to have several outliers when controlling for magnesium. Medians don't differ a lot, indigating, that magnesium alone may not be a good discriminator.  
Heatmap shows that some features clearly correlate (positively or negatively)

PCA: 
Dimensionality reduction and scaling of the features allows for idenification of clusters and patterns. 
Wine classes cluster well together which indicates their similarity in the feature variables. 
Still, some outliers can be identified. 

Important features: 
LME models suggest that most important features to predict the wine class are proline concentration, the OD and the flavanoids concentration.

A research hypothesis could therefore be: 
H1: "The proline concentration is different between different wine classes and can be used as a distinguisher factor."
H0:"The proline concentration is equal between the wine classes." 

Accuracy of the models: 
Accuracy assessment of the models for training and test data reveal, that the regression model is very accurate (100% for the training data and 97% for the test data.) This indicates that wineclasses can be described easily by approximately linear boundaries in the standardized feature space and logistic regression can perform well. 
However, the decision tree accuracy is lower (39% and 36%). Depth=3 limits the complexity and may be to shallow to distinguish classes using a decision tree. 