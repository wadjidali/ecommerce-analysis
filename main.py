import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

path = kagglehub.dataset_download("hellbuoy/online-retail-customer-clustering")
df = pd.read_csv(path + "/OnlineRetail.csv", encoding='ISO-8859-1')

df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month

top_products = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_products.plot(kind='bar')
plt.title("Top produits")
plt.savefig("images/top_products.png")
plt.show()

monthly_sales = df.groupby('Month')['TotalPrice'].sum()

plt.figure()
monthly_sales.plot()
plt.title("Ventes mensuelles")
plt.savefig("images/monthly_sales.png")
plt.show()

