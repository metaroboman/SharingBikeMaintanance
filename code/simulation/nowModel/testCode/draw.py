import pandas as pd
import matplotlib.pyplot as plt

add = 'C:\\Rebalancing\\nowModel\\test\\testGetPerformance.csv'
df = pd.read_csv(add, names=['a', 'b'])
plt.plot(df.a, df.b)
plt.show()
