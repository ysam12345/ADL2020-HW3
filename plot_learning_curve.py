import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
with open('plot/pg_learning_curve.json') as f:
    learning_curve = json.load(f)



data = {}
data['episode'] = []
data['reward'] = []
for key, value in learning_curve.items():
    data['episode'].append(key)
    data['reward'].append(value)
df = pd.DataFrame(data)


for i in range(0,df.shape[0]-19):
    df.loc[df.index[i+19],'SMA_20'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1]
                                                +df.iloc[i+3,1]+ df.iloc[i+4,1] +df.iloc[i+5,1]
                                                +df.iloc[i+6,1]+ df.iloc[i+7,1] +df.iloc[i+8,1]
                                                +df.iloc[i+9,1]+ df.iloc[i+10,1] +df.iloc[i+11,1]
                                                +df.iloc[i+12,1]+ df.iloc[i+13,1] +df.iloc[i+14,1]
                                                +df.iloc[i+15,1]+ df.iloc[i+16,1] +df.iloc[i+17,1]
                                                +df.iloc[i+18,1]+ df.iloc[i+19,1])/20),1)
df['pandas_SMA_20'] = df.iloc[:,1].rolling(window=20).mean()

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df['reward'],label='average reward')
plt.plot(df['pandas_SMA_20'],label='SMA 20 average reward')
plt.legend(loc=2)
plt.title('Learning Curve of Reward (window = 20)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("pg_learning_curve.png")

plt.show()

