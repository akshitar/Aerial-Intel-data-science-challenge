import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

def initializeCol(df, col):
    df[col] = pd.Series()
    return df[col]

def fill_Nans(df, value):
    return df.fillna(value=value, axis='columns')

def fillwithAverage(df, columnName, cNum):
    for num in cNum:
        print num
        data = df[df.clusterNum==num]
        for i, row in enumerate(data.index):
            if i==1:
                df[columnName].loc[row] = np.mean([data[columnName].loc[data.index[i-1]], data[columnName].loc[data.index[i]]])
            elif i>=2:
                if (data[columnName].loc[row]==0):
                    df[columnName].loc[row] = np.mean([data[columnName].loc[data.index[i-2]], data[columnName].loc[data.index[i-1]]])
                else:
                    df[columnName].loc[row] = np.mean([data[columnName].loc[data.index[i-2]], data[columnName].loc[data.index[i-1]], data[columnName].loc[data.index[i]]])
    return df[columnName]

def returnCol(df):
    return df.columns

def plotScatter(x,y):
    plt.scatter(x,y)
    plt.show()

def pltCommands(xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def returnDummyVar(column):
    data_columnName = pd.get_dummies(column)
    return data_columnName

def returnPlot(df, column1, column2Name, names):
    #color = ["r-","b-","g-","y-","c-","m-","k-"]
    for name, col in zip(names,colors.cnames):
        data = df[column1==name]
        mean_Value = []
        dates = np.unique(data.Date)
        for date in dates:
            mean_Value.append(np.mean(data[column2Name][data.Date==date]))

        plt.plot_date(x=dates, y=mean_Value,fmt=col,linestyle='-',label=name)

    plt.legend(loc='upper left', frameon=True)
    plt.title("Aggregate Yield over the year 2013-2014 and 2014-2015")
    pltCommands('date',column2Name)

def getMeanValues(df, column):
    mean_val = []
    state_name = np.unique(df.State)
    for name in state_name:
        data = df[df.State==name]
        mean_val.append(np.mean(data[column]))

    return mean_val, state_name

def returnBarPlot(df1, df2, column):
    firstYear_val, state_name = getMeanValues(df1 ,column)
    secondYear_val, state_name = getMeanValues(df2 ,column)
    width = 0.35
    ind = np.arange(len(state_name))
    p1 = plt.bar(ind, firstYear_val, width, color='g')
    p2 = plt.bar(ind+width, secondYear_val, width, color='m')
    plt.xticks(ind + width/2., (state_name))
    plt.legend((p1[0], p2[0]), ('2013 Year Data', '2014 Year Data'))
    pltCommands(column,'')

def returnPreviousCols(df, columnName, cNum ):
    df['dummyt2'] = initializeCol(df, 'dummyt2')
    df['dummyt1'] = initializeCol(df, 'dummyt1')
    for num in cNum:
        print num
        data = df[df.clusterNum==num]
        for i, row in enumerate(data.index):
            if i>=2:
                df['dummyt2'].loc[row] = data[columnName].loc[data.index[i-2]]
                df['dummyt1'].loc[row] = data[columnName].loc[data.index[i-1]]
            else:
                df['dummyt2'].loc[row] = data[columnName].loc[data.index[0]]
                df['dummyt1'].loc[row] = data[columnName].loc[data.index[0]]
    return df['dummyt2'], df['dummyt1']

def create_correlation_matrix(df, array):
    corr_matrix = np.zeros((len(array), len(array)))
    for i,coli in enumerate(array):
        for j,colj in enumerate(coli[i:]):
            if i != j:
                corr_matrix[i, j] = np.corrcoef(df.loc[:, coli].T,df.loc[:, colj].T)[0, 1]
                corr_matrix[j, i] = corr_matrix[i, j]
            else:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
    return corr_matrix


df1 = pd.read_csv('/Users/Akshita/Downloads/wheat-2013-supervised.csv')
df2 = pd.read_csv('/Users/Akshita/Downloads/wheat-2014-supervised.csv')

# Fill Nans with zeros
df1 = fill_Nans(df1,0)
df2 = fill_Nans(df2,0)

# Join the two dataframes
frames = [df1, df2]
df = pd.concat(frames).reset_index(drop=True)

# Statistics of the data
cols = returnCol(df)
for i in range(0,df.shape[1],6):
    print df[cols[i:i+6]].info()

# Identify relationship between features
cols = returnCol(df) - ['Date','Longitude','Latitude','CountyName','State','Yield','precipTypeIsOther']
print create_correlation_matrix(df, cols)

# Relation between independent and dependent variable
cols = returnCol(df) - ['Date','Longitude','Latitude','CountyName','State','Yield','precipTypeIsOther']
for i in cols:
    plotScatter(df[i], df['Yield'])

# Histograms of features
cols = returnCol(df) - ['Date','Longitude','Latitude','CountyName','State','Yield','precipTypeIsOther']
f, ax = plt.subplots(5, sharex=True)
for i in range(0,5):
    print cols[i+15]
    ax[i].hist(df[cols[i+15]], bins=10)
    ax[i].set_xlabel(cols[i+15])
plt.show()

# Convert to Datetime
df.Date = pd.to_datetime(df.Date)

# Average Yield trend for every State
names = np.unique(df.State)
returnPlot(df, df.State, 'Yield', names)

# Bar plot for average conditions for every year
cols = returnCol(df) - ['Date','Longitude','Latitude','CountyName', 'State', 'precipTypeIsOther']
returnBarPlot(df1, df2, 'cloudCover')

# Separating train and test data randomly
cols = returnCol(df) - ['Date','CountyName', 'State', 'Yield', 'NDVI']
X_train, X_test, y_train, y_test = train_test_split(df[cols], df.Yield, test_size=0.33, random_state=4)

# Build Baseline model
reg_model = ensemble.GradientBoostingRegressor(loss='ls', max_depth=5, n_estimators= 500,learning_rate=0.01)
reg_model = reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
names = df.columns.values
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), reg_model.feature_importances_), names),reverse=True)
print "Validation error", mean_squared_error(y_test, y_pred)
print "Training error", mean_squared_error(reg_model.predict(X_train), y_train)

# Predicted vs Actual values of Yield
y_test = np.array(y_test)
plotScatter(y_test,y_pred)

# Scatter plot of Lat/Long for every state
names = np.unique(df.State)
color = ['red','green','blue','cyan','yellow']
for name, col in zip(names,color):
    data = df[df.State==name]
    plt.scatter(data.Latitude, data.Longitude, c=col, label=name)

plt.legend(loc='upper left', frameon=True)
plt.title("Scatter plot")
pltCommands('Latitude',"Longitude")

# Clustering of data-points
est = KMeans(n_clusters=5, random_state=56)
y_pred = est.fit_predict(df[['Latitude','Longitude']])
plt.scatter(df.Latitude, df.Longitude, c=y_pred)
plt.title("Clustering plot")
pltCommands('Latitude',"Longitude")

# Insert the cluster number as a feature
# 0==Texas, 1==Montana, 2==Kansas, 3==Washington, 4==Oklahoma
df['clusterNum'] = initializeCol(df, 'clusterNum')
df['clusterNum'] = y_pred

# Convert precipIntensity into categorical
df.precipCategorical = initializeCol(df, 'precipCategorical')
for row in range(0,df.shape[1]):
    if df.precipIntensity.loc[row] < 0.002:
        df.precipCategorical = 'No Precip'
    elif df.precipIntensity.loc[row] < 0.017:
        df.precipCategorical = 'Very light'
    elif df.precipIntensity.loc[row] < 0.1:
        df.precipCategorical = 'Light'
    elif df.precipIntensity.loc[row] < 0.4:
        df.precipCategorical = 'Moderate'
    else:
        df.precipCategorical = 'Heavy'

# Remove pressure and visibility and put in the average over T-2, T-1 and T
cNum = np.unique(y_pred)
df.pressure = fillwithAverage(df, 'pressure',cNum)
df.visibility = fillwithAverage(df, 'visibility', cNum)

# Get features for T-2 and T-1
col = ['windSpeed', 'windBearing', 'temperatureMin', 'temperatureMax', 'dewPoint']
for c in col:
    dummyt2 = str(c)+ str('t-2')
    dummyt1 = str(c)+ str('t-1')
    df[dummyt2] = initializeCol(df, dummyt2)
    df[dummyt1] = initializeCol(df, dummyt1)
    df[dummyt2], df[dummyt1] = returnPreviousCols(df, c, cNum)

# Save variable
Y = df.Yield

# Convert 'precipCategorical' using one-hot encoding
column_precip = returnDummyVar(df['precipCategorical'])
df = df.join([column_precip])

# Drop columns
col = ['CountyName','State','Date','precipTypeIsOther','precipTypeIsSnow','precipTypeIsRain','apparentTemperatureMax','apparentTemperatureMin',
'precipIntensity','precipIntensityMax','precipAccumulation', 'precipProbability', 'NDVI','Yield','precipCategorical']
for i in col:
    df = df.drop(i, axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.33, random_state=4)

# Use cross-validation to select parameters of model
param = {'max_depth':range(5,10,2), 'min_samples_split':range(200,800,200), 'n_estimators':range(20,51,10),'min_samples_leaf':range(30,61,10), 'max_features':range(7,13,2)}
gsearch = GridSearchCV(estimator = ensemble.GradientBoostingRegressor(learning_rate=0.1, subsample=0.8, random_state=10),
param_grid = param, scoring="mean_squared_error" ,n_jobs=4,iid=False, cv=5)
gsearch.fit(X_train,y_train)
print gsearch.best_params_

# With the best paprametrs build the GBM
para = gsearch.best_params_
gbm_tuned = ensemble.GradientBoostingRegressor(**para)
gbm_tuned.fit( X_train, y_train)
y_pred = gbm_tuned.predict(X_test)

# Predicted vs Actual values of Yield
y_test = np.array(y_test)
plotScatter(y_test,y_pred)
print "Validation error", mean_squared_error(y_test, y_pred)
print "Training error", mean_squared_error(gbm_tuned.predict(X_train), y_train)

