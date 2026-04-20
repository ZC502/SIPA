import matplotlib.pyplot as plt
import pandas as pd

# log data
data = {
    "time": [1.2, 1.4, 1.6, 2.0, 2.2, 2.4, 2.6],
    "assoc": [0, 0.4, -2.8, 43.5, 3090.7, 11224.2, 5.1]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["assoc"], color='red', linewidth=2, marker='o')
plt.yscale('log') # logarithmic coordinate
plt.title("SIPA Diagnostic: IRB 14050 Redundancy Jump Detection")
plt.xlabel("Time (s)")
plt.ylabel("NARH Associator Score (Log Scale)")
plt.grid(True, which="both", ls="-", alpha=0.5)

# comment
plt.annotate('Critical Discontinuity', xy=(2.4, 11224.2), xytext=(1.8, 15000),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
