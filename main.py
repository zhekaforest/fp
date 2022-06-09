import pandas as pd
import matplotlib.pyplot as plt
#get Amsterdam airbnb database
df = pd.read_csv("listings.csv")
print(df.columns)

plt.hist(df.price, range=(0, 1000), bins=100, cumulative=False)
plt.xlabel("price")
plt.show()








