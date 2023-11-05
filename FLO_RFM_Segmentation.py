
###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to divide its customers into segments and determine marketing strategies according to these segments.
# Towards this end, customers' behaviors will be defined and groups will be created based on these behavioral clusters...


# The data set consists of information obtained from the past shopping behavior of customers who made their last purchases
# via OmniChannel (both online and offline shopping) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel : Channel where last purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The customer's last purchase date on the online platform
# last_order_date_offline : The last shopping date of the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform   F
# order_num_total_ever_offline : Total number of purchases made by the customer offline   F
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases  M
# customer_value_total_ever_online : Total fee paid by the customer for online shopping  M
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
df.head()
###############################################################
###############################################################

# Data Understanding and preparation

import datetime as dt
import pandas as pd
pd.options.display.width = 1000
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/ödevler/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()

df.head(10)

df.columns

df.describe().T

df.isnull().sum()


df.dtypes

df['order_num_total'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

df.groupby('master_id').agg({'order_num_total': 'sum',
                            'customer_value_total': 'sum'})

# Examine the variable types. Change the type of variables expressing date to date.
# first way
df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
df["last_order_date_offline"]= df["last_order_date_offline"].astype("datetime64[ns]")

# second way
for i in df.columns:
    if "date" in i:
        df[i]=df[i].apply(pd.to_datetime)

df.dtypes


# Look at the distribution of the number of customers, average number of products purchased and average expenditures
# across shopping channels.

df.groupby("order_channel").agg({"master_id": ["count"],
                                 "order_num_total" : ["mean"],
                                  "customer_value_total": ["mean"]}).head()


# List the top 10 customers that bring the most profit.

df.sort_values("customer_value_total", ascending=False)[:10]


df.groupby(["master_id"]).agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)

# List the top 10 customers who place the most orders.

df.sort_values("order_num_total", ascending=False)[:10]

# second way
df.groupby(["master_id"]).agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)

# Functionalize the data preparation process.
def create_func(df):
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
    df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
    df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
    df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")
    return df

df_yeni= create_func(df)
df_yeni.dtypes

# Calculating RFM Metrics

# Recency   = "last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days
# The analysis date is 2 days after the date of the last purchase in the data set.
df["last_order_date"].max() # Which day was the last order placed?
today_date = dt.datetime(2021,6,1) # Analysis day was chosen 2 days after the last day
type(today_date)

df.head()

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_num_total': lambda order_num_total: order_num_total.sum(),
                                     'customer_value_total': lambda customer_value_total: customer_value_total.sum()})


# Instead of groupby it could have been like this
#rfm = pd.DataFrame()
#rfm["customer"] = df["master_id"]
#rfm["frequency"] = df["order_num_total"]
#rfm["recency"] = (today_date - df["last_order_date"]).dt.days  # dt.days direkt değişken üstüne uygulayınca ,.days lambdaya olunca
#rfm["monetary"] = df["total_price"]


rfm.columns = ["Recency", "Frequency", "Monetary"]


# Calculation of RF and RFM Scores
rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"]= (rfm["recency_score"].astype(str)+
                  rfm["frequency_score"].astype(str))

# Defining RF Scores as Segments
# RFM assignment

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# replace is a string method.
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)



# Examine the recency, frequency and monetary averages of the segments.
rfm.groupby('segment').agg({'Recency': 'mean',
                           'Frequency': 'mean',
                           'Monetary': 'mean'})

rfm[rfm["RFM_SCORE"] == "45"]



# Functionalize the entire process.

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_ = pd.read_csv("/Users/melihasecilozturk/Desktop/miuul/ödevler/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()
df.head()

def create_rfm(dataframe, csv=False):

    # Data Preparation
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
    df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
    df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
    df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")

    # Determination RFM metrics
    today_date = dt.datetime(2021, 6, 1)  # son günden 2 gün sonrası analiz günü seçildi


    rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                       'order_num_total': lambda order_num_total: order_num_total.sum(),
                                       'customer_value_total': lambda customer_value_total: customer_value_total.sum()})

    rfm.columns = ["Recency", "Frequency", "Monetary"]


    # Determination RFM scores
    rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])


    # cltv_df scores were converted to categorical values ​​and added to df
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # SEGMENT ASSIGNMENT

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["Recency", "Frequency", "Monetary", "segment"]]


    if csv:
        rfm.to_csv("rfm_analiz.csv")

    return rfm


rfm_new = create_rfm(df, csv=True)




















