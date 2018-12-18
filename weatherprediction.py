#%%
import pandas as pd
import numpy as np
import math
from math import sqrt
from numpy import linalg

#--------------***********************----------------#
#--------------*** obtain coordinate ***--------------#
##read air_quality coordinates
excel_path="/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/beijing_airquality_station.xlsx"
air=pd.read_excel(excel_path)
air.rename(columns={"Station ID":"station_id_x"},inplace=True)

##read grid coordinates
grid=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/Beijing_grid_weather_station.csv",header=-1)
grid.columns=["ID","latitude1","longitude1"]
cols = list(grid)
cols.insert(1,cols.pop(cols.index('longitude1')))
grid = grid.loc[:,cols]

##read observe coordinates
observe=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/observedWeather_201701-201801.csv")
obser=observe.drop_duplicates("station_id")[['station_id','longitude','latitude']]
obser=obser.reset_index(drop=True)

#--------------***********************--------------------------------------------------------------#
#--------------*** compute distances and pair the minimal three locations with air ***--------------#
##compute the distance between air and grid
##compute the distance between air and observed data
def find(x):
    l3=[]
    for indexs in air.index:
        l1=[]
        l2=[]
    #print(air.loc[indexs].values[0:3])
        vector1=np.array(air.loc[indexs].values[1:3])
        for i in x.index:
            vector2=np.array(x.loc[i].values[1:3])
            l1.append(np.linalg.norm(vector1-vector2))
            j=l1.index(min(l1))
       
        l2.append(air.loc[indexs].values[0:1][0])
        l2.append(air.loc[indexs].values[1:2][0])
        l2.append(air.loc[indexs].values[2:3][0])
        l2.append(x.loc[j].values[0:1][0])
        l2.append(x.loc[j].values[1:2][0])
        l2.append(x.loc[j].values[2:3][0])
        l2.append(min(l1))
        l3.append(l2)
    return l3

of=find(obser)
gf=find(grid)

final=[]
for i in range(0,35):
    if gf[i][0]==of[i][0]:
        if gf[i][6]<of[i][6]:
            final.append(gf[i])
        elif gf[i][6]>of[i][6]:
            final.append(of[i])
        else:
            final.append(gf[i])
final=pd.DataFrame(final)

lg=[]
for indexs in air.index:
    l1=[]
    l2=[]
    l2.append(air.loc[indexs].values[0:1][0])
    #print(air.loc[indexs].values[0:3])
    vector1=np.array(air.loc[indexs].values[1:3])
    for i in grid.index:
        vector2=np.array(grid.loc[i].values[1:3])
        
        if np.linalg.norm(vector1-vector2)<0.1 and len(l1)<=1:
            l1.append(np.linalg.norm(vector1-vector2))
            l2.append(grid.loc[i].values[0:1][0])
            l2.append(grid.loc[i].values[1:2][0])
            l2.append(grid.loc[i].values[2:3][0])
    lg.append(l2)
lg=pd.DataFrame(lg)
finall=pd.merge(final,lg,on=0)

del finall['1_x']
del finall['2_x']
del finall['4_x']
del finall['5_x']
del finall['6_x']
del finall['2_y']
del finall['3_y']
del finall['5_y']
del finall['6_y']

finall.rename(columns={0:'station_id','3_x':'station1',
                       '1_y':'station2','4_y':'station3'},inplace=True)


#merge tables--grid
grid1=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/gridWeather_201701-201803.csv")
grid2=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/gridWeather_201804.csv")
del grid2['id']
del grid1['longitude']
del grid1['latitude']
grid1.rename(columns={'stationName':'station_id', 'utc_time':'time',
                      'wind_speed/kph':'wind_speed'}, inplace = True)
col_name = grid1.columns.tolist()
col_name.insert(2,'weather')
grid1=grid1.reindex(columns=col_name)
gridall = pd.concat([grid1,grid2],axis=0,sort=False)
gridall['wind_speed']=gridall['wind_speed']/3.6
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars=gridall['weather']
gridall['weather'] = le.fit_transform(cat_vars.tolist())
gridall['weather']=gridall['weather'].replace(9,0)

#finall['station2'].unique()
#finall['station3'].unique()
#gridfinal2.columns.values.tolist()
gridfinal1=gridall[gridall['station_id'].isin(['beijing_grid_303','beijing_grid_282','beijing_grid_304',
                                  'beijing_grid_263','beijing_grid_262','beijing_grid_239',
                                  'beijing_grid_261','beijing_grid_238','beijing_grid_301',
                                  'beijing_grid_323','beijing_grid_366','beijing_grid_240',
                                  'beijing_grid_265','beijing_grid_224','beijing_grid_414',
                                  'beijing_grid_452','beijing_grid_385','beijing_grid_278',
                                  'beijing_grid_216'])]
gridfinal2=gridall[gridall['station_id'].isin(['beijing_grid_303', 'beijing_grid_302', 'beijing_grid_282',
                                   'beijing_grid_281', 'beijing_grid_283', 'beijing_grid_262',
                                   'beijing_grid_242', 'beijing_grid_261', 'beijing_grid_239',
                                   'beijing_grid_240', 'beijing_grid_238', 'beijing_grid_301',
                                   'beijing_grid_322', 'beijing_grid_345', 'beijing_grid_347',
                                   'beijing_grid_264', 'beijing_grid_452', 'beijing_grid_349',
                                   'beijing_grid_391', 'beijing_grid_203', 'beijing_grid_413',
                                   'beijing_grid_364', 'beijing_grid_278', 'beijing_grid_215'])]
gridfinal3=gridall[gridall['station_id'].isin(['beijing_grid_304', 'beijing_grid_303', 'beijing_grid_283',
                                   'beijing_grid_282', 'beijing_grid_262', 'beijing_grid_263',
                                   'beijing_grid_281', 'beijing_grid_240', 'beijing_grid_261',
                                   'beijing_grid_239', 'beijing_grid_302', 'beijing_grid_323',
                                   'beijing_grid_365', 'beijing_grid_348', 'beijing_grid_265',
                                   'beijing_grid_241', 'beijing_grid_453', 'beijing_grid_350',
                                   'beijing_grid_392', 'beijing_grid_204', 'beijing_grid_223',
                                   'beijing_grid_414', 'beijing_grid_473', 'beijing_grid_385',
                                   'beijing_grid_279', 'beijing_grid_216', 'beijing_grid_324'])]

gridfinal2.rename(columns={'weather':'weather2','temperature':'temperature2',
                           'pressure':'pressure2','humidity':'humidity2',
                           'wind_direction':'wind_direction2','wind_speed':'wind_speed2'},inplace=True)
gridfinal3.rename(columns={'weather':'weather3','temperature':'temperature3',
                           'pressure':'pressure3','humidity':'humidity3',
                           'wind_direction':'wind_direction3','wind_speed':'wind_speed3'},inplace=True)

#merge tables--observed
obser1=pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/observedWeather_201701-201801.csv')
obser2=pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/observedWeather_201802-201803.csv')
obser3=pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/observedWeather_201804.csv')

del obser3['id']
obser3.rename(columns={'time':'utc_time'}, inplace = True)
obser2.insert(1,'longitude',None)
obser2.insert(2,'latitude',None)
obserall=pd.concat([obser1,obser2,obser3], axis=0, join='outer', sort=False,ignore_index=True) 

obserfinal=obserall[obserall['station_id'].isin(['chaoyang_meo','hadian_meo','fengtai_meo',
                                                'shunyi_meo','pingchang_meo','pinggu_meo',
                                                'huairou_meo','miyun_meo','yanqing_meo'])]

del obserfinal['longitude']
del obserfinal['latitude']
obserfinal.columns=['station_id','time','temperature','pressure','humidity','wind_direction','wind_speed','weather']

obserfinal[obserfinal.isnull().values==True]
obserfinal.insert(1,'date',obserfinal['time'])
obserfinal.insert(1,'hour',obserfinal['time'])
obserfinal["date"]=obserfinal["time"].map(lambda x:x.split()[0])
obserfinal["hour"]=obserfinal["time"].map(lambda x:x.split()[1])
obserfinal.sort_values(by=['station_id','hour'],ascending=True,inplace=True)
obserfinal['wind_direction']=obserfinal['wind_direction'].fillna(method='pad')
obserfinal['wind_speed']=obserfinal['wind_speed'].fillna(method='pad')
obserfinal.sort_values(by=['station_id','time'],ascending=True,inplace=True)

#merge tables--air quality
target1 = pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/airQuality_201701-201801.csv')
target2 = pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/airQuality_201802-201803.csv')
target3 = pd.read_csv('/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/aiqQuality_201804.csv')

del target3['id']
target3.rename(columns={'station_id':'stationId', 'time':'utc_time',
                       'PM25_Concentration':'PM2.5', 'PM10_Concentration':'PM10',
                       'NO2_Concentration':'NO2', 'CO_Concentration':'CO',
                       'O3_Concentration':'O3', 'SO2_Concentration':'SO2'}, inplace = True)

target =pd.concat([target1,target2,target3],axis=0,sort=False)
target =target.replace(' ', np.nan)
target['CO'] = target['CO'].fillna(method='pad')
target['NO2'] = target['NO2'].fillna(method='pad')
target['O3'] = target['O3'].fillna(method='pad')
target['PM10'] = target['PM10'].fillna(method='pad')
target['PM2.5'] = target['PM2.5'].fillna(method='pad')
target['SO2'] = target['SO2'].fillna(method='pad')

target.rename(columns={'stationId':'station_id','utc_time':'time'}, inplace = True)


#merge three tables into air quality
f=finall[['station_id','station1','station2','station3']]
he=pd.merge(target,f,on='station_id')

obserfinal['stt1']=obserfinal['station_id']+obserfinal['time']
gridfinal1['stt1']=gridfinal1['station_id']+gridfinal1['time']
gridfinal2['stt2']=gridfinal2['station_id']+gridfinal2['time']
gridfinal3['stt3']=gridfinal3['station_id']+gridfinal3['time']
he['stt1']=he['station1']+he['time']
he['stt2']=he['station2']+he['time']
he['stt3']=he['station3']+he['time']

a=pd.merge(he,obserfinal,on='stt1')
b=pd.merge(he,gridfinal1,on='stt1')
p=pd.concat([a,b], axis=0,sort=False,ignore_index=True)
p.rename(columns={'station_id_x':'station_id1','time_x':'time'},inplace=True)
c=pd.merge(p,gridfinal2,on='stt2')
d=pd.merge(c,gridfinal3,on='stt3')


d=d.fillna(-1000)
air.rename(columns={"station_id_x":"station_id1"},inplace=True)
train=pd.merge(d,air,on='station_id1')

train=train[['time','PM2.5','PM10','NO2','CO','O3','SO2','temperature','pressure',
             'humidity','wind_direction','wind_speed','weather','weather2','temperature2',
             'pressure2','humidity2','wind_direction2', 'wind_speed2','weather3',
             'temperature3','pressure3','humidity3','wind_direction3','wind_speed3',
             'longitude','latitude']]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars=train['weather']
train['weather'] = le.fit_transform(cat_vars.tolist())
cat_vars1=train['time']
train['time'] = le.fit_transform(cat_vars1.tolist())

#--------------***********************-------------#
#--------------*** Model Training ***--------------#
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
X=train[['longitude','latitude','time', 'temperature','pressure','humidity','wind_direction',
         'wind_speed','weather','weather2','temperature2','pressure2','humidity2',
         'wind_direction2','wind_speed2','weather3','temperature3','pressure3','humidity3',
         'wind_direction3','wind_speed3','NO2','SO2','CO']]
scaler = StandardScaler() 
scaler.fit(X)  
XS = scaler.transform(X)

Y=train[['PM2.5','PM10','O3']]

Y1=train['PM10']
Y2=train['PM2.5']
Y3=train['O3']
YNO2=train['NO2']
YSO2=train['SO2']
YCO=train['CO']
YY=[['NO2','SO2','CO']]
YY1=train['NO2']
YY2=train['SO2']
YY3=train['CO']
#X['new']=X['NO2']*X['SO2']*X['CO']
def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    c = np.array(actual) + np.array(predicted)
    denominator =c
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=(denominator!=0), casting='unsafe'))

#--------------*** XGboost Algorithm + Grid Search + Cross Validator ***********-------------#
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import     RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
X_train, X_test, y1_train, y1_test = train_test_split(XS, Y1, test_size=0.3, random_state=0)
param_test = {'max_features': [2,5,8],'max_depth': [40,80,100]}
xb1 = xgb.XGBRegressor(subsample=0.6,n_estimators=200,min_child_weight=0.03,random_state=50)
xb1.fit(X_train, y1_train)
grid_search1 = GridSearchCV(xb1,param_test,cv=5)
grid_search1.fit(X_train, y1_train)
x1p=np.abs(grid_search1.predict(X_test))
print('xg1',smape(y1_test,x1p))
print("Test set score:{:.2f}".format(grid_search1.score(X_test,y1_test)))
print("Best parameters:{}".format(grid_search1.best_params_))
print("Best score on train set:{:.2f}".format(grid_search1.best_score_))
#xg1 0.25931631006826994
#Test set score:0.78
#Best parameters:{'max_depth': 40, 'max_features': 2, 'random_state': 50}
#Best score on train set:0.79

X_train, X_test, y2_train, y2_test = train_test_split(XS, Y2, test_size=0.3, random_state=0)
print("Parameters:{}".format(param_test))
xb = xgb.XGBRegressor(subsample=0.6,n_estimators=200,min_child_weight=0.03,random_state=50)
grid_search2 = GridSearchCV(xb,param_test,cv=5)
grid_search2.fit(X_train, y2_train)
x2p=np.abs(grid_search2.predict(X_test))
print('xg2',smape(y2_test,x2p))
print("Test set score:{:.2f}".format(grid_search2.score(X_test,y2_test)))
print("Best parameters:{}".format(grid_search2.best_params_))
print("Best score on train set:{:.2f}".format(grid_search2.best_score_))
#xg2 0.24049361245135686
#Test set score:0.90
#Best parameters:{'max_depth': 40, 'max_features': 2, 'random_state': 50}
#Best score on train set:0.89

X_train, X_test, y3_train, y3_test = train_test_split(XS, Y3, test_size=0.3, random_state=0)
xb3 = xgb.XGBRegressor(subsample=0.6,n_estimators=200,min_child_weight=0.03,random_state=50)
xb3.fit(X_train, y3_train)
grid_search3 = GridSearchCV(xb3,param_test,cv=5)
grid_search3.fit(X_train, y3_train)
x3p=np.abs(grid_search3.predict(X_test))
print('xg3',smape(y2_test,x3p))
print("Test set score:{:.2f}".format(grid_search3.score(X_test,y3_test)))
print("Best parameters:{}".format(grid_search3.best_params_))
print("Best score on train set:{:.2f}".format(grid_search3.best_score_))
#xg3 0.3861947130978906
#Test set score:0.91
#Best parameters:{'max_depth': 40, 'max_features': 2, 'random_state': 50}
#Best score on train set:0.90


#--------------**************** Lightgbm+RFECV***********-------------------------------#
import lightgbm as lgb
X_train, X_test, y1_train, y1_test = train_test_split(X, Y1, test_size=0.3, random_state=0)
gbm1 = lgb.LGBMRegressor(objective='regression',num_leaves=300,
                              learning_rate=0.1, n_estimators=500, max_depth=40, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=19,min_child_weight=0.001,
                              bagging_fraction=0.8, feature_fraction=0.8) 

rfecv = RFECV(estimator=gbm1, step=1, cv=5)
rfecv.fit(X_train, y1_train)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print(rfecv.support_)
print(rfecv.ranking_)
X_t=[]
mask = rfecv.get_support() 
new_features = X.columns[mask]
for x in new_features:
    X_t.append(X[x])
X_t=pd.DataFrame(X_t)
X_t=X_t.T
X_train, X_test, y1_train, y1_test = train_test_split(X_t, Y1, test_size=0.3, random_state=0)
gbm1.fit(X_train,y1_train)
g1p=np.abs(gbm1.predict(X_test))
print('g1p',smape(y1_test,g1p ))
#%%
X_train, X_test, y2_train, y2_test = train_test_split(X, Y2, test_size=0.3, random_state=0)
gbm2 = lgb.LGBMRegressor(objective='regression',num_leaves=300,
                              learning_rate=0.1, n_estimators=500, max_depth=40, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=19,min_child_weight=0.001,bagging_fraction=0.8, feature_fraction=0.8) 

rfecv = RFECV(estimator=gbm2, step=1, cv=3)
rfecv.fit(X_train, y2_train)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print(rfecv.support_)
print(rfecv.ranking_)
X_t=[]
mask = rfecv.get_support() #list of booleans[3 1 1 1 1 1 2 1 4]
new_features = X.columns[mask]
for x in new_features:
    X_t.append(X[x])
X_t=pd.DataFrame(X_t)
X_t=X_t.T
X_train, X_test, y2_train, y2_test = train_test_split(X_t, Y2, test_size=0.3, random_state=0)
gbm2.fit(X_train,y2_train)
g2p=np.abs(gbm2.predict(X_test))
print('g2p',smape(y2_test,g2p ))
#%%
X_train, X_test, y3_train, y3_test = train_test_split(X, Y3, test_size=0.3, random_state=0)
gbm3 = lgb.LGBMRegressor(objective='regression',num_leaves=300,
                              learning_rate=0.1, n_estimators=500, max_depth=40, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=19,min_child_weight=0.001,bagging_fraction=0.8, feature_fraction=0.8) 

rfecv = RFECV(estimator=gbm3, step=1, cv=3)
rfecv.fit(X_train, y3_train)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print(rfecv.support_)
print(rfecv.ranking_)
X_t=[]
mask = rfecv.get_support() #list of booleans[3 1 1 1 1 1 2 1 4]
new_features = X.columns[mask]
for x in new_features:
    X_t.append(X[x])
X_t=pd.DataFrame(X_t)
X_t=X_t.T
X_train, X_test, y3_train, y3_test = train_test_split(X_t, Y3, test_size=0.3, random_state=0)
gbm3.fit(X_train,y3_train)
g3p=np.abs(gbm3.predict(X_test))
print('g3p',smape(y3_test,g3p ))
#%%
#--------------***********************---------#
#--------------*** test data ***--------------#
##process predictive data
gridpre1=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/gridWeather_20180501-20180502.csv")
gridpre2=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/gridWeather_20180501-20180502.csv")
gridpre3=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/gridWeather_20180501-20180502.csv")
obserpre=pd.read_csv("/home/joker/MSBD5002PROJECT_data/MSBD5002PROJECT_data/observedWeather_20180501-20180502.csv")

del gridpre1['id']
del gridpre2['id']
del gridpre3['id']
gridpre1['wind_speed']=gridpre1['wind_speed']/3.6
gridpre2['wind_speed']=gridpre2['wind_speed']/3.6
gridpre3['wind_speed']=gridpre3['wind_speed']/3.6
del obserpre['id']

gridpre2.rename(columns={'weather':'weather2','temperature':'temperature2',
                           'pressure':'pressure2','humidity':'humidity2',
                           'wind_direction':'wind_direction2','wind_speed':'wind_speed2'},inplace=True)
gridpre3.rename(columns={'weather':'weather3','temperature':'temperature3',
                           'pressure':'pressure3','humidity':'humidity3',
                           'wind_direction':'wind_direction3','wind_speed':'wind_speed3'},inplace=True)


import time
import datetime

timelist=[]
timelist=np.array(gridpre1['time'].unique())

#timelist
p=[]
for i in timelist:
    p.append(i)
    

new1=[]
for index in air['station_id1']:
    for j in range(0,48):
        new1.append(index)
new1=pd.DataFrame(new1)

new2=[]
for j in range(0,35):
    for time in p:
        new2.append(time)
new2=pd.DataFrame(new2)

new1['id']=new1.index
new2['id']=new2.index
new=pd.merge(new1,new2,on='id')
new.rename(columns={'0_x':'station_id','0_y':'time'},inplace=True)
del new['id']

he1=pd.merge(new,f,on='station_id')

obserpre['st1']=obserpre['station_id']+obserpre['time']
gridpre1['st1']=gridpre1['station_id']+gridpre1['time']
gridpre2['st2']=gridpre2['station_id']+gridpre2['time']
gridpre3['st3']=gridpre3['station_id']+gridpre3['time']

he1['st1']=he1['station1']+he1['time']
he1['st2']=he1['station2']+he1['time']
he1['st3']=he1['station3']+he1['time']

a1=pd.merge(he1,obserpre,on='st1')
b1=pd.merge(he1,gridpre1,on='st1')
p1=pd.concat([a1,b1], axis=0,sort=False,ignore_index=True,join='outer')
p1.rename(columns={'station_id_x':'station_id1','time_x':'time'},inplace=True)
c1=pd.merge(p1,gridpre2,on='st2')
d1=pd.merge(c1,gridpre3,on='st3')

air.rename(columns={"station_id_x":"station_id1"},inplace=True)
test=pd.merge(d1,air,on='station_id1')
test=test[['station_id1','longitude','latitude','time', 'temperature','pressure','humidity','wind_direction',
         'wind_speed','weather','weather2','temperature2','pressure2','humidity2',
         'wind_direction2','wind_speed2','weather3','temperature3','pressure3','humidity3',
         'wind_direction3','wind_speed3']]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars2=test['weather']
cat_vars3=test['weather2']
cat_vars4=test['weather3']
test['weather'] = le.fit_transform(cat_vars2.tolist())
test['weather2'] = le.fit_transform(cat_vars3.tolist())
test['weather3'] = le.fit_transform(cat_vars4.tolist())


test['time_x']=test['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
cat_vars5=test['time_x']
test['time_x'] = le.fit_transform(cat_vars5.tolist())

Z=test[['longitude','latitude','time_x', 'temperature','pressure','humidity','wind_direction',
         'wind_speed','weather','weather2','temperature2','pressure2','humidity2',
         'wind_direction2','wind_speed2','weather3','temperature3','pressure3','humidity3',
         'wind_direction3','wind_speed3']]
Z.rename(columns={"time_x":"time"},inplace=True)
#%%
XX=train[['longitude','latitude','time', 'temperature','pressure','humidity','wind_direction',
         'wind_speed','weather','weather2','temperature2','pressure2','humidity2',
         'wind_direction2','wind_speed2','weather3','temperature3','pressure3','humidity3',
         'wind_direction3','wind_speed3']]

from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
scaler = StandardScaler() 
scaler.fit(XX)  
XXS = scaler.transform(XX)
scaler = StandardScaler() 
scaler.fit(Z)  
ZS = scaler.transform(Z)


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, yno2_train, yno2_test = train_test_split(XXS, YNO2, test_size=0.3, random_state=0)
gbm1 = lgb.LGBMRegressor(objective='regression',learning_rate=0.1, n_estimators=500, 
                        metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8,
                        min_child_samples=19, min_child_weight=0.001,num_leaves=300,max_depth=40) 
gbm1.fit(X_train, yno2_train)
yno2_p=gbm1.predict(ZS)
X_train, X_test, yso2_train, yso2_test = train_test_split(XXS, YSO2, test_size=0.3, random_state=0)
gbm2 = lgb.LGBMRegressor(objective='regression',learning_rate=0.1, n_estimators=500, 
                        metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8,
                        min_child_samples=19, min_child_weight=0.001,num_leaves=300,max_depth=40) 
gbm2.fit(X_train,  yso2_train) 
yso2_p=gbm2.predict(ZS)
X_train, X_test, yco_train, yco_test = train_test_split(XXS, YCO, test_size=0.3, random_state=0)
gbm3 = lgb.LGBMRegressor(objective='regression',learning_rate=0.1, n_estimators=500, 
                        metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8,
                        min_child_samples=19, min_child_weight=0.001,num_leaves=300,max_depth=40) 
gbm3.fit(X_train, yco_train)
yco_p=gbm3.predict(ZS)
yno2_p=np.abs(yno2_p)
yso2_p=np.abs(yso2_p)
yco_p=np.abs(yco_p)
yno2_p=pd.DataFrame(yno2_p)
yso2_p=pd.DataFrame(yso2_p)
yco_p=pd.DataFrame(yco_p)
Z.reset_index(inplace=True,drop=True)
Z1=pd.concat([Z,yno2_p,yso2_p,yco_p],axis=1)
scaler = StandardScaler() 
scaler.fit(Z1) 
ZZ = scaler.transform(Z1)

#%%

from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(XXS, YY, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=30,n_estimators=400,random_state=10)
rf.fit(X_train, y_train)
result=rf.predict(Z)
result=pd.DataFrame(result)
Z.reset_index(inplace=True, drop=True)
ans=pd.concat([Z,result], axis=1 )
ans.rename(columns={0:'NO2',1:'SO2',2:'CO'}, inplace = True)




#--------------***********************---------#
#--------------*** Prediction ***--------------#

#--------------**************************** XGboost ****************************-------------#
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import     RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb

X_train, X_test, y1_train, y1_test = train_test_split(XS, Y1, test_size=0.3, random_state=20)
xb1 = xgb.XGBRegressor(max_depth=40,max_features=2,random_state=50,subsample=0.6,n_estimators=200,min_child_weight=0.03)
xb1.fit(X_train, y1_train)
x1p=np.abs(xb1.predict(X_test))
print(smape(y1_test, x1p))
#0.2035532360951042   


from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
X_train, X_test, y2_train, y2_test = train_test_split(XS, Y2, test_size=0.3, random_state=20)
xb2 = xgb.XGBRegressor(max_depth=40,max_features=6,random_state=50,subsample=0.6,n_estimators=200,min_child_weight=0.03)
xb2.fit(X_train, y2_train)
x2p=np.abs(xb2.predict(X_test))
print(smape(y2_test, x2p))
#0.20348091435416554  

from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
X_train, X_test, y3_train, y3_test = train_test_split(XS, Y3, test_size=0.3, random_state=20)
xb3 = xgb.XGBRegressor(max_depth=40,max_features=2,random_state=50,subsample=0.6,n_estimators=200,min_child_weight=0.03)
xb3.fit(X_train, y3_train)
x3p=np.abs(xb3.predict(X_test))
print(smape(y3_test, x3p))

x1p=pd.DataFrame(x1p)
x2p=pd.DataFrame(x2p)
x3p=pd.DataFrame(x3p)
x1p.reset_index(inplace=True,drop=True)
xp=pd.concat([x1p,x2p,x3p],axis=1)
y_test=pd.concat([y1_test,y2_test,y3_test],axis=1)
print('xg',smape(y_test,xp))

x1pt=np.abs(xb1.predict(Z1))
x2pt=np.abs(xb2.predict(Z1))
x3pt=np.abs(xb3.predict(Z1))
x1pt=pd.DataFrame(x1pt)
x2pt=pd.DataFrame(x2pt)
x3pt=pd.DataFrame(x3pt)
x1pt.reset_index(inplace=True,drop=True)
xpt=pd.concat([x1pt,x2pt,x3pt],axis=1)
xpt.columns = ['PM10','PM2.5','O3']
xpt.to_csv('xpt.csv')
#0.20466450682556145
#0.20462119287357286
#0.23462802618423423
#xg 0.21463790862778953

#--------------**************** Lightgbm***********--------------------------------------------------#

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y1_train, y1_test = train_test_split(XS, Y1, test_size=0.3, random_state=0)
gbm11 = lgb.LGBMRegressor(objective='regression',learning_rate=0.1, n_estimators=500, 
                        metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8,
                        min_child_samples=19, min_child_weight=0.001,num_leaves=300,max_depth=40) 
gbm11.fit(X_train, y1_train) 
y_pred = gbm11.predict(X_test) # 模型评估 
gb11=np.abs(y_pred)
print('gbm1',smape(y1_test,gb11))


X_train, X_test, y2_train, y2_test = train_test_split(XS, Y2, test_size=0.3, random_state=0)
gbm22 = lgb.LGBMRegressor(objective='regression',num_leaves=300,
                              learning_rate=0.1, n_estimators=500, max_depth=40, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=19,
                              min_child_weight=0.001,bagging_fraction=0.8, feature_fraction=0.8) 
gbm22.fit(X_train, y2_train) 
y_pred = gbm22.predict(X_test) # 模型评估 
gb22=np.abs(y_pred)
print('gbm2',smape(y2_test,gb22))
X_train, X_test, y3_train, y3_test = train_test_split(XS, Y3, test_size=0.3, random_state=0)
gbm33 = lgb.LGBMRegressor(objective='regression',learning_rate=0.1, n_estimators=500, 
                        metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8,
                        min_child_samples=19, min_child_weight=0.001,num_leaves=300,max_depth=40) 
gbm33.fit(X_train, y3_train) 
y_pred = gbm33.predict(X_test) # 模型评估 
gb33=np.abs(y_pred)
print('gbm3',smape(y3_test,gb33))
gb11=pd.DataFrame(gb11)
gb22=pd.DataFrame(gb22)
gb33=pd.DataFrame(gb33)
gb11.reset_index(inplace=True,drop=True)
gb44=pd.concat([gb11,gb22,gb33],axis=1)
y_test=pd.concat([y1_test,y2_test,y3_test],axis=1)
print('gbm',smape(y_test,gb44))
#gbm1 0.23873675082216225
#gbm2 0.228891151245295
#gbm3 0.2746548298065363
#gbm 0.24742757729133105
gb1t=np.abs(gbm11.predict(ZZ))
gb2t=np.abs(gbm22.predict(ZZ))
gb3t=np.abs(gbm33.predict(ZZ))
gb1t=pd.DataFrame(gb1t)
gb2t=pd.DataFrame(gb2t)
gb3t=pd.DataFrame(gb3t)
gb1t.reset_index(inplace=True,drop=True)
gbt=pd.concat([gb1t,gb2t,gb3t],axis=1)
#--------------**************************** RandomForest ****************************-------------#
ans
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(XS, Y, test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(max_depth=35,n_estimators=400,random_state=10)
rf1.fit(X_train, y_train)
rft=rf1.predict(ans)
scaler = StandardScaler() 
scaler.fit(ans)  
ans = scaler.transform(ans)
#%%

#--------------****************************** Output *********************************-------------#

from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder

ID=test[['station_id1','time']]
gbt1=pd.concat([ID,gbt],axis=1)
gbt1['num']=gbt1['time'].map(lambda x:x.split( )[1]).map(lambda x:x.split(':')[0])
gbt1['num']=LabelEncoder().fit_transform(gbt1['num'])
d_index=list(gbt1.columns).index('num')
print(d_index)
print(gbt1.iloc[1,d_index])
for i in range(0, len(gbt1)):
    if gbt1.iloc[i]['time'].split( )[0].split('-')[2]=='02':
        gbt1.iloc[i,d_index]=gbt1.iloc[i,d_index]+24
gbt1['num']=gbt1['num'].astype(str)
print(gbt1.dtypes)   
gbt1['test_id']=gbt1['station_id1']+'#'+gbt1['num']
test_id=pd.DataFrame(gbt1[['test_id','station_id1','time']])
gbt1.sort_values(by=['station_id1','time'],ascending=True,inplace=True)


rft.rename(columns={'0':'PM2.5','1':'PM10','2':'O3'},inplace=True)
gbt.rename(columns={'0':'PM10','0.1':'PM2.5','0.2':'O3'},inplace=True)
predict=pd.DataFrame(data=[],columns=['PM2.5','PM10','O3'])
predict['PM2.5']=rft['PM2.5']+gbt['PM2.5']+xpt['PM2.5']
predict['PM2.5']=predict['PM2.5']/3
predict['PM10']=rft['PM10']+gbt['PM10']+xpt['PM10']
predict['PM10']=predict['PM10']/3
predict['O3']=rft['O3']+gbt['O3']+xpt['O3']
predict['O3']=predict['O3']/3
predict=pd.concat([test_id,predict],axis=1)
predict.sort_values(by=['station_id1','time'],ascending=True,inplace=True)
predict=predict[['test_id','PM2.5','PM10','O3']]
predict.to_csv('predict.csv')


df=predict

del df['Unnamed: 0']

df.loc[len(df)] = {'test_id':'dongsihuan_aq#29','PM2.5':47.89435930389159,
                   'PM10':84.422701583973,'O3':145.9730082636608}
df.loc[len(df)] ={'test_id':'dongsihuan_aq#28','PM2.5':42.84793959581101,
                  'PM10':87.53207664249732,'O3':125.57369752685452}
df.loc[len(df)] ={'test_id':'fengtaihuayuan_aq#28','PM2.5':101.59519781958666,
                  'PM10':305.14291408471456,'O3':91.6468184258323}
df.loc[len(df)] ={'test_id':'fengtaihuayuan_aq#29','PM2.5':42.87381298031143,
                  'PM10':79.35654917094027,'O3':140.74168760664855}
df.loc[len(df)] ={'test_id':'huairou_aq#28','PM2.5':104.45118502829396,
                  'PM10':114.04184616403754,'O3':72.1628380729649}
df.loc[len(df)] ={'test_id':'huairou_aq#29','PM2.5':38.65697393236248,
                  'PM10':91.31137103579368,'O3':86.6777061610858}
df.loc[len(df)] ={'test_id':'miyun_aq#28','PM2.5':49.571101032910775,
                  'PM10':49.571101032910775,'O3':172.68166292504714}
df.loc[len(df)] ={'test_id':'miyun_aq#29','PM2.5':37.40342654179689,
                  'PM10':81.07158614398963,'O3':129.19797587549246}
df.loc[len(df)] ={'test_id':'nongzhanguan_aq#28','PM2.5':40.54477392555416,
                  'PM10':56.150955198903844,'O3':144.14900413459895}
df.loc[len(df)] ={'test_id':'nongzhanguan_aq#29','PM2.5':36.05325009272185,
                  'PM10':36.05325009272185,'O3':127.18265198976826}
df.loc[len(df)] ={'test_id':'pingchang_aq#28','PM2.5':42.08061544094935,
                  'PM10':67.48069383114264,'O3':150.86626652165242}
df.loc[len(df)] ={'test_id':'pingchang_aq#29','PM2.5':63.54632202941612,
                  'PM10':87.57733672435015,'O3':154.45095207992117}
df.loc[len(df)] ={'test_id':'pinggu_aq#28','PM2.5':67.52460935128892,
                  'PM10':79.59570908136358,'O3':168.01687621828765}
df.loc[len(df)] ={'test_id':'pinggu_aq#29','PM2.5':82.74477241122345,
                  'PM10':82.74477241122345,'O3':174.00210291065346}
df.loc[len(df)] ={'test_id':'shunyi_aq#28','PM2.5':41.235787270187025,
                  'PM10':54.06365691364288,'O3':156.6650395307979}
df.loc[len(df)] ={'test_id':'shunyi_aq#29','PM2.5':15.64970438108345,
                  'PM10':70.53107556092162,'O3':99.00614818795091}
df.loc[len(df)] ={'test_id':'wanliu_aq#28','PM2.5':43.54711823114788,
                  'PM10':59.66811201415079,'O3':143.7797296160753}
df.loc[len(df)] ={'test_id':'wanliu_aq#29','PM2.5':39.129085572441625,
                  'PM10':89.60007232122152,'O3':148.17017979452635}
df.loc[len(df)] ={'test_id':'xizhimenbei_aq#28','PM2.5':31.91689471185829,
                  'PM10':64.55860575867743,'O3':127.59407620452512}
df.loc[len(df)] ={'test_id':'xizhimenbei_aq#29','PM2.5':38.95305231500446,
                  'PM10':75.19895295399624,'O3':128.0381660703613}
df.loc[len(df)] ={'test_id':'yanqin_aq#28','PM2.5':49.465273874775605,
                  'PM10':114.9700169804294,'O3':180.13424518074095}
df.loc[len(df)] ={'test_id':'yanqin_aq#29','PM2.5':82.35839610067706,
                  'PM10':148.41329504330417,'O3':184.9652273527165}

p['station']=p['test_id'].map(lambda x:x.split('#')[0])
p['time']=p['test_id'].map(lambda x:x.split('#')[1])
p['time']=p['time'].astype(int)
p.sort_values(by=['station','time'],ascending=True,inplace=True)
p=p[['test_id','PM2.5','PM10','O3']]
p.to_csv('p.csv')
del p['Unnamed: 0']
result = pd.merge(submission, p, how='left', on=['test_id'])
del result['PM2.5_x']
del result['PM10_x']
del result['O3_x']
result.rename(columns={'PM2.5_y':'PM2.5','PM10_y':'PM10','O3_y':'O3'},inplace=True)
result.to_csv('submission.csv')



