import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load and inspect
df = pd.read_csv('quickcart_shipments.csv')
print(df.shape)
print(df.info())
print(df.isnull().sum())

# 2. Handle missing values
num_cols = ['distance_km', 'package_weight_kg', 'delivery_cost']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['customer_zone'] = df['customer_zone'].fillna(df['customer_zone'].mode()[0])
df.dropna(subset=['order_id', 'hub_id'], inplace=True)

# 3. Outlier detection and treatment (IQR method)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers_iqr(df, 'delivery_cost')
df = remove_outliers_iqr(df, 'distance_km')

# 4. Remove duplicates and standardize formats
df.drop_duplicates(subset='order_id', keep='last', inplace=True)
df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
df['customer_zone'] = df['customer_zone'].str.strip().str.lower()

# 5. Normalization
scaler = MinMaxScaler()
scale_cols = ['distance_km', 'package_weight_kg', 'delivery_cost']
df[scale_cols] = scaler.fit_transform(df[scale_cols])
