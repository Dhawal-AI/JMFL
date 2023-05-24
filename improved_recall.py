# %%
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn import preprocessing
from google.cloud import bigquery
from sklearn.utils import resample
from sklearn.utils import class_weight
from datetime import datetime, timedelta
from scipy.stats import chi2_contingency
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics  import accuracy_score, recall_score,confusion_matrix, precision_score,roc_auc_score, roc_curve
from sklearn import metrics
import pytz
import os
import glob
from google.cloud import storage
import warnings
import pandas as pd
import numpy as np
import os
import glob
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.cloud import storage
import json
# from sklearn.ensemble import CatBoostClassifier
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.knn import KNN 
from pyod.utils.data import generate_data, get_outliers_inliers
warnings.simplefilter("ignore")

# %%

print("------------------------------------------------------------------")
fmt = "%Y-%m-%d %H:%M:%S"
now_utc = datetime.now(pytz.timezone('UTC'))
print("------------------------------------------------")
print("UTC time------", now_utc.strftime(fmt))
print(
    "start --------------------",
    now_utc.astimezone(
        pytz.timezone('Asia/Kolkata')).strftime(fmt))

seed = 100

client = bigquery.Client()

print("Client creating using default project: {}".format(client.project))
query = """
  
  with curr_data as (
  SELECT * FROM `jm-ebg.JM_PredictN_Production.jm_predictN_cur_session_his_agg_data_esd`
),
cpd_data as (

  SELECT GAID_Cr, ProspectID_LeadCreation, Source_LeadUpdate as source, 
  SourceMedium_LeadUpdate as medium, SourceCampaign_LeadUpdate as campaign, 
  mx_AOF_Stage_LeadUpdate as mx_AOF_Stage, mx_DIY_Stage_LeadUpdate as mx_DIY_Stage,EmailAddress_LeadUpdate as email,
  Phone_LeadUpdate as Phone,ProspectActivityName_Max_LeadUpdate as ProspectActivityName_Max, 
  mx_Last_Activity_done_by_Agent_LeadUpdate as mx_Last_Activity_done_by_Agent, 
  mx_Sub_Dispostion_LeadUpdate  as mx_Sub_Dispostion from `jm-ebg.cdp_staging.LSQ-LeadMaster`
)

select *except(GAID_Cr, ProspectID_LeadCreation) from curr_data
left join (select * from cpd_data)
on ProspectId = ProspectID_LeadCreation
"""
query_job = client.query(
    query,
    # Location must match that of the dataset(s) referenced in the query.
    # location="US",
)  # API request - starts the query
df = query_job.to_dataframe()

#print("\nchanged df")
#print(df.loc[df["ad_group"]=="%7Badgroupid%7D"])
data=pickle.load(open('/home/jupyter/jm_predictN/Tatvic_approach/production/model/cat_dict_esd_RF.pkl', 'rb'))
model = pickle.load(open('/home/jupyter/jm_predictN/Dhawal/model/Dhawal_v1.pkl', 'rb'))
### list of column name dtype wise
string_column = list(df.select_dtypes(include=['O']).columns)
int_column = list(df.select_dtypes(include=['int64']).columns)
float_column = list(df.select_dtypes(include=['float64']).columns)
cat_cols = list(df.select_dtypes(include=['O']).columns)

# df['placement'].loc[df['placement'].apply(lambda x: (x not in data['placement']) )] = '(Others)'
df['device_category_past'].loc[df['device_category_past'].apply(lambda x: (x not in data['device_category_past']) )] = '(Others)'
df['mobile_brand_name_past'].loc[df['mobile_brand_name_past'].apply(lambda x: (x not in data['mobile_brand_name_past']) )] = '(Others)'
df['operating_system_past'].loc[df['operating_system_past'].apply(lambda x: (x not in data['operating_system_past']) )] = '(Others)'
df['os_version_past'].loc[df['os_version_past'].apply(lambda x: (x not in data['os_version_past']) )] = '(Others)'
df['browser_past'].loc[df['browser_past'].apply(lambda x: (x not in data['browser_past']) )] = '(Others)'
df['city_past'].loc[df['city_past'].apply(lambda x: (x not in data['city_past']) )] = '(Others)'
df['region_past'].loc[df['region_past'].apply(lambda x: (x not in data['region_past']) )] = '(Others)'
df['ft_campaign_past'].loc[df['ft_campaign_past'].apply(lambda x: (x not in data['ft_campaign_past']) )] = '(Others)'
df['ft_medium_past'].loc[df['ft_medium_past'].apply(lambda x: (x not in data['ft_medium_past']) )] = '(Others)'
df['ft_source_past'].loc[df['ft_source_past'].apply(lambda x: (x not in data['ft_source_past']) )] = '(Others)'
df['utm_term_placement_past'].loc[df['utm_term_placement_past'].apply(lambda x: (x not in data['utm_term_placement_past']) )] = '(Others)'
# df['placement_past'].loc[df['placement_past'].apply(lambda x: (x not in data['placement_past']) )] = '(Others)''

### replace the string int and float value
for i in string_column:
    df[i].fillna("Value_Not_Found", inplace = True)
for i in int_column:
    df[i].fillna(-99999, inplace = True) 
for i in float_column:
    df[i].fillna(-99999999.000, inplace = True) 

# %%
df = df.drop_duplicates(keep='first').reset_index(drop=True)
predict_info=df[['ProspectId', 'user_pseudo_id', 'visitStartTime', 'event_date','email',
                 'Phone','source', 'medium', 'campaign']]
# %%
drop_cols = ['ProspectId', 'user_pseudo_id', 'visitStartTime', 'event_date','device_category', 'ad_network', 'ad_group', 'ProspectID_Past', 'user_pseudo_id_past', 'visitStartTime_past', 'date_past', 'timeOnSite_past', 'day_of_week_past', 'device_category_past', 'mobile_brand_name_past', 'operating_system_past', 'os_version_past', 'browser_past', 'region_past', 'city_past', 'ft_campaign_past', 'ft_medium_past', 'ft_source_past', 'ad_network_past', 'ad_group_past', 'utm_term_placement_past', 'source', 'medium', 'campaign', 'mx_AOF_Stage', 'mx_DIY_Stage', 'email', 'Phone']
df = df.drop(drop_cols, axis=1)
#print(df)
cat_cols = list(df.select_dtypes(include=['O']).columns)
#print(cat_cols)
# c_l=['ad_group','ad_group_past']
# cat_cols = cat_cols.drop(c_l, axis=1)
# %%

# cat_cols = cat_cols.remove("ad_group")
# cat_cols_= cat_cols.remove("ad_group_past")
#print(cat_cols)
 # le = pickle.load(open("/home/jupyter/jm_predictN/Tatvic_approach/production/label_encode/{}-esd-RF.pkl".format(i), 'rb'))

# print(cat_cols)
#categorical value handling
le = LabelEncoder()
for  i in cat_cols:
    df[i] = le.fit_transform(df[i])
    pickle.dump(le, open("/home/jupyter/jm_predictN/Dhawal/label_encode/{}-esd-RF.pkl".format(i), 'wb'))
## datatype as training
# feature_dtype = pd.read_csv('/home/jupyter/jm_predictN/Tatvic_approach/production/model/col_dtypes_esd_RF.csv')
# feature_type = {}
# for idx, rows in feature_dtype.iterrows():
#     df[rows['column_name']] = df[rows['column_name']].astype(rows['dtypes'])
#     feature_type[rows['column_name']] = rows['dtypes']

# cat_columns = list(df.dtypes[df.dtypes == 'O'].index)
# print (cat_columns)
# all_columns = list(df.columns)
# cat_idxs = []
# for column in cat_columns:
#     cat_idxs.append(all_columns.index(column))
# print(df.info())
# print(feature_type)
# df = df.astype(feature_type)
# df = df[['timeOnSite', 'device_category', 'mobile_brand_name',
#        'operating_system', 'os_version', 'region', 'city', 'ft_medium',
#        'ft_source', 'sw_medium', 'sw_source', 'mobile_entered',
#        'mobile_verified', 'timeOnSite_past', 'device_category_past',
#        'mobile_brand_name_past', 'operating_system_past', 'os_version_past',
#        'region_past', 'city_past', 'ft_medium_past', 'ft_source_past',
#        'sw_medium_past', 'sw_source_past', 'mobile_entered_past',
#        'mobile_verified_past']]
# drop_cols = ['ad_network','ad_network_past']
#'ProspectId' 'user_pseudo_id' 'visitStartTime' 'event_date'\n 'ProspectID_Past' 'user_pseudo_id_past' 'visitStartTime_past' 'date_past'\n 'user_psudo_id_conv' 'ProspectId_conv' 'date_conv' 'ad_group'\n 'ad_group_past'
# df['ad_group']=df.ad_group.replace('',np.nan).astype(float)

# df["device_category_past"]=df['device_category_past'].astype(float)# df = df.drop(drop_cols, axis=1)
df.rename(columns={'timeOnSite': 'timeOnSite_past', 'day_of_week': 'day_of_week_past','device_category': 'device_category_past','mobile_brand_name': 'mobile_brand_name_past','operating_system': 'operating_system_past','browser': 'browser_past','region': 'region_past','city': 'city_past','ft_campaign': 'ft_campaign_past','ft_medium': 'ft_medium_past','ft_source': 'ft_source_past','utm_term_placement': 'utm_term_placement_past','os_version':'os_version_past'}, inplace=True)
df=df.loc[:,['timeOnSite_past', 'day_of_week_past', 'mobile_brand_name_past',
       'operating_system_past', 'os_version_past', 'browser_past',
       'region_past', 'city_past', 'ft_campaign_past', 'ft_medium_past',
       'ft_source_past', 'utm_term_placement_past',
        'ProspectActivityName_Max',
       'mx_Last_Activity_done_by_Agent', 'mx_Sub_Dispostion']]


ss_train = MinMaxScaler()
df= ss_train.fit_transform(df)
preds_train = model.predict_proba(df)
predict_info['score'] = preds_train[:, 1]
prediction_datetime = (now_utc.astimezone(pytz.timezone('Asia/Kolkata'))).strftime(fmt)
print("date of prediction --", prediction_datetime)

#Decile threshold
lower_threshold=None
upper_threshold=None
# %%
if lower_threshold==None and upper_threshold==None:
    lower_thresh = predict_info['score'].quantile(0.3)
    higher_thresh = predict_info['score'].quantile(0.7)
else:
    lower_thresh = float(lower_threshold)
    higher_thresh = float(upper_threshold)

# %%
print ("Lower Threshold = ", lower_thresh)
print ("Upper Threshold = ", higher_thresh)


# %%


# create bucket
predict_info['bucket'] = np.where(predict_info['score'] >= higher_thresh, 1, 0)
predict_info['bucket'] = np.where(((predict_info['score'] >= lower_thresh) & (
            predict_info['score'] < higher_thresh)),2,predict_info['bucket'])
predict_info['bucket'] = np.where(predict_info['score'] <= lower_thresh,3,
    predict_info['bucket'])
# predict_info['prediction_datetime'] = prediction_datetime
predict_info['product_category'] = 'JM_PredictN'

try : 
    pred_date_bg_tablename = (
            now_utc.astimezone(
                pytz.timezone('Asia/Kolkata'))).strftime("%Y%m%d")
    query = """
        SELECT * FROM `jm-ebg.JM_PredictN_Production.jm_predictN_prediction_esd_{}`  
        """.format(pred_date_bg_tablename)


    query_job = client.query(
            query
        )  

    bq_df_exclude = query_job.to_dataframe()
except : 
    bq_df_exclude = pd.DataFrame(columns =["ProspectId", "user_pseudo_id", "visitStartTime", "event_date","email","Phone", "score", "bucket", "product_category",'source', 'medium', 'campaign', 'mx_AOF_Stage', 'mx_DIY_Stage'])

print("bq_df_exclude shape",bq_df_exclude.shape)
prediction_df_bg = predict_info.sort_values('score', ascending=False)
# prediction_df_bg = prediction_df_bg.drop_duplicates(
#     subset=['ProspectId','event_date'],
#     keep='first').reset_index(
#     drop=True)

print("unique data of prospectid ---", prediction_df_bg.shape)
if not bq_df_exclude.empty:
    prediction_df_bg = pd.concat([prediction_df_bg, bq_df_exclude]).reset_index(drop=True)
    prediction_df_bg = prediction_df_bg.drop_duplicates(subset=['ProspectId','event_date'],keep=False).reset_index(drop=True)
    ### only consider real time data
    # prediction_df_bg = prediction_df_bg.loc[prediction_df_bg['prediction_datetime'] == prediction_datetime].reset_index(drop=True)
else:
    print('Empty BQ dataframe')
    pass


print("exclude previous session---",prediction_df_bg.shape)
prediction_df_bg['event_date']  = prediction_df_bg['event_date'].astype('str')
print("dataframe shape bigquery", prediction_df_bg.shape)
print("Bigquery intertion start")
if not prediction_df_bg.empty:
    # timedelta(days=5) timedelta(minutes=30)
    pred_date_bg_tablename = str(predict_info.event_date.unique()[0])
    # frist method - days wise table create
    # print("prediction_df_bg", prediction_df_bg.shape())
    bq_client = bigquery.Client()
    dataset_ref = bq_client.dataset('Hyperparameter_Tuned_Prediction')
    job_config = bigquery.LoadJobConfig(schema=[
        bigquery.SchemaField(name="ProspectId", field_type="STRING"),
        bigquery.SchemaField(name="user_pseudo_id", field_type="STRING"),
        bigquery.SchemaField(name="visitStartTime", field_type="INTEGER"),
        bigquery.SchemaField(name="event_date", field_type="STRING"),
        bigquery.SchemaField(name="email", field_type="STRING"),
        bigquery.SchemaField(name="Phone", field_type="STRING"),
        bigquery.SchemaField(name="source", field_type="STRING"),
        bigquery.SchemaField(name="medium", field_type="STRING"),
        bigquery.SchemaField(name="campaign", field_type="STRING"),
        bigquery.SchemaField(name="score", field_type="FLOAT"),
        bigquery.SchemaField(name="bucket", field_type="INTEGER"),
        # bigquery.SchemaField(name="prediction_datetime",field_type="STRING"),
        bigquery.SchemaField(name="product_category", field_type="STRING")
    ])
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED

    table_ref = dataset_ref.table("JM_PRED_HP_RF_"+pred_date_bg_tablename)
    dataframe = pd.DataFrame(prediction_df_bg.to_records(index=False))
    job = bq_client.load_table_from_dataframe(
        dataframe, table_ref, job_config=job_config)
    job.result()  # Waits for table load to complete.
else:
    pass
print("Bigquery intertion end")
