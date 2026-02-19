import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv('machine_data.csv')
print(df)

print(df.shape)

grpByManu = df.groupby('manufacturef')

dfa = grpByManu.get_group('A')
dfb = grpByManu.get_group('B')
dfc = grpByManu.get_group('c')

print("\n--- Range per manufacturer ---")
for name, d in [("A", dfa), ("B", dfb), ("C", dfc)]:
    print(
        f"{name}: "
        f"load min={d['load'].min():.3f}, max={d['load'].max():.3f} | "
        f"time min={d['time'].min():.3f}, max={d['time'].max():.3f}"
    )

loada = dfa['load']
timea = dfa['time']

loadb = dfb['load']
timeb = dfb['time']

loadc = dfc['load']
timec = dfc['time']

print("\n--- Median load per manufacturer ---")
print("A:", dfa["load"].median())
print("B:", dfb["load"].median())
print("C:", dfc["load"].median())

plt.figure()
plt.scatter(loada, timea, label='A', s=10)
plt.scatter(loadb, timeb, label='B', s=10)
plt.scatter(loadc, timec, label='C', s=10)
plt.title("Relation between load and time (A/B/C)")
plt.xlabel("Load")
plt.ylabel("Time")
plt.legend()
plt.show()

dfa['load'].mean()

dfa[['load']].plot(kind='hist', bins=10)

plt.figure()
plt.hist(loada, bins=10, alpha=0.6, label='A')
plt.hist(loadb, bins=10, alpha=0.6, label='B')
plt.hist(loadc, bins=10, alpha=0.6, label='C')
plt.title("Load distribution (A/B/C)")
plt.xlabel("Load")
plt.ylabel("Count")
plt.legend()
plt.show()

plt.figure()
plt.hist(timea, bins=10, alpha=0.6, label='A')
plt.hist(timeb, bins=10, alpha=0.6, label='B')
plt.hist(timec, bins=10, alpha=0.6, label='C')
plt.title("Time distribution (A/B/C)")
plt.xlabel("Time")
plt.ylabel("Count")
plt.legend()
plt.show()

plt.figure()
plt.boxplot([timea, timeb, timec], tick_labels=['A','B','C'])
plt.title("Time by manufacturer (boxplot)")
plt.ylabel("Time")
plt.show()