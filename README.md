# Aerial-Intel-data-science-challenge

Goal:
With two years of data we want to predict wheat yield for several counties across the United States given geographical features, meteorological features and knowledge of season.

Timeline:
08/18 - 08/19 : Understand the dataset's structure using statistical summaries and data visualization. Both dataset had missing values for 'pressure' and 'visibility'. They were filled with zeros initally. The statistical summaries are a good way to get a range of data and also eyeball for outliers.  Next, was to get the inter-correlation between features and how features vary with 'Yield'. NDVI had high correlations with 'temperatureMax', 'apparenttemperatureMax' and 'precipIntensity' and hence was dropped. Visualizations of the data included generating their histograms to get a distribution of the features ,the average trend of 'Yield' for every State over two years and also how the average values of features vary over the years.

08/20 - 08/23 : Get some background knowledge about the features that are given. For example: how ApparentTemp differs from Temp, how is NDVI calculated, how are windBearing and windSpeed related, is pressure/visibilty related to any other atmospheric parameters that are given. Next using the raw data but excluding 'Date', 'State' and 'CountyName', built GradientBoosted model to get a baseline model which had a slight overfit and also to understand the contribution of each feature to prediction. Latitude and Longitude were the most relevant and this was the basis for the next model. I wanted to incorporate the knowledge of 'State' feature implicitly and the scatter plot of Latitide/Longitude showed how the datapoints for every State were clustered tightly together. I used KMeans for clustering the data-points together and the cluster number was used as a feature. In order to deal with features pressure and visibility that have missing values, I repaced them with their moving averages t-2, t-1 and t. 'precipIntensity' was converted to a categorical varaibles. The purpose behind this was to incorporate all the necessary information by dropping all other all other 'precip' related feature (and also feature importances from GBM model for all other 'precip' related features were very low). 'apparentTemp' features were dropped. For features 'windSpeed' 'windBearing' 'temperatureMin' 'temperatureMax' 'dewPoint' their previous known values t-2 and t-1 were also added and this was done according to their cluster number. The intention behind this was to have the previous values State-wise and not let previous values of one State interfere with the other. These features were feeded to a GBM model and as before MSE was chosen as the performance measure. Finally, the scatter plot of Actual vs Predicted values which is a good means of visulazing how our model is performing such that for good performance most points should be close to a regressed diagonal line a relative improvement confirms the improvement over the baseline model. 

Final approach:
My final approach involved the above feature engineered attributes and building a GBM on this data with a goal to optimize the Mean Squared error. It performed better than the baseline model with a 46% relative decrease in MSE.

Technical choices:
The first choice was to use a GB model and the main motivation behind this was the non-linear relationship of features with 'Yield'. It has a nice by-product of feature importances which is always useful to gain inights. The motivation to use KMeans was only that I had an idea of how many clusters actually exists. I also used cross-validation to prune the parameters of the final model. 

Challenges:
My biggest challenge was my lack of knowledge of this domain. I think feature engineering is an integral part of any model and in order to get features that describe the structures inherent in our data some domain knowledge is indeed necessary. 

What did you learn along the way?:
My experience in Machine Learning has mostly been with classification problems so this whole process was enlightning and challenging for me. I started the assignment a little too cautiously but as the project advanced I trusted my inutition and reasoning to undertake the various decisions across this project.

Future work:
I would definitely try to improve clustering. I carried out KMeans on only Latitude/Longitude but there can be other features with more discriminating power able to further refine the clusters. I could not improve on my final model but I could build other models and ensemble them to get better results.
