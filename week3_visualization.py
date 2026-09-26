import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Exploratory analysis
summary = df[['delivery_time_hr', 'distance_km', 'weight_kg', 'delivery_cost']].describe()
corr = df[['delivery_time_hr', 'distance_km', 'weight_kg', 'delivery_cost']].corr()
print(summary)
print(corr)

# Distribution of delivery times
sns.histplot(df['delivery_time_hr'], bins=30, kde=True)
plt.title('Distribution of Delivery Times')
plt.show()

# Delivery cost by hub
sns.boxplot(data=df, x='hub', y='delivery_cost')
plt.title('Delivery Cost by Hub')
plt.show()

# Correlation heatmap
sns.heatmap(corr, annot=True, cmap='Oranges')
plt.title('Correlation Between Key Variables')
plt.show()

# Monthly shipment volume trend
monthly = df.groupby('month').size()
sns.lineplot(x=monthly.index, y=monthly.values, marker='o')
plt.title('Monthly Shipment Volume Trend')
plt.show()

# Cost vs distance scatter
sns.scatterplot(data=df, x='distance_km', y='delivery_cost', hue='hub')
plt.title('Delivery Cost vs. Distance')
plt.show()
