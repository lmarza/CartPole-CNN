# check this https://www.youtube.com/watch?v=GcXcSZ0gQps&t=2096s for more details

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_style("darkgrid")
approch = pd.read_csv('e2e.csv')
ax = sns.lineplot(x="Episode", y="Score", data=approch, ci= 'sd')
plt.title('DQN CartPole-v1')
plt.xlabel('No of episodes')
plt.ylabel('Score (t)')
plt.legend(loc='lower right')
plt.show(ax)

