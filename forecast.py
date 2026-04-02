import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
 
data ={
     "Months":[1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12],
    "Sales":[15000, 18000, 16000, 22000, 25000, 24000, 28000, 30000, 32000, 27000, 35000, 38000 ],

}

df=pd.DataFrame(data)

x=df[["Months"]]
y=df[["Sales"]]
model=LinearRegression()
model.fit(x,y)

future_months=pd.DataFrame({"Months":[13,14,15]})
predictions=model.predict(future_months)

for i,pred in zip([13,14,15],predictions):
    print(f"Months {i} predicted sales: {pred[0]:,.0f}")
plt.figure(figsize=(10,5))
plt.plot(df["Months"],df["Sales"],color="blue",marker="o",label="Actual Sales")
plt.plot([12, 13, 14, 15], [38000, 38739, 40309, 42239], color="red",linestyle="--",marker="o",label="Predicted Sales")
plt.title("Sales Forecast")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png")
print("Chart saved!")


   