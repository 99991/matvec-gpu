import matplotlib.pyplot as plt
import numpy as np
import json

with open("data/results.json") as f:
    results = json.load(f)

# Sort by min time
names = np.array(list(results.keys()), dtype=object)
min_times = np.array([min(results[name]) for name in names])
indices = np.argsort(min_times)[::-1]
names = names[indices]
min_times = min_times[indices]

print("| Framework | Min Time [µs] | Median [µs] | Max [µs] |")
print("| --- | --- | --- | --- |")
for name in names[::-1]:
    times = results[name]
    print(f"| [{name}](https://github.com/99991/matvec-gpu/blob/main/{name}) | {min(times):.3f} | {np.median(times):.3f} | {max(times):.3f} |")

# Plot min times for each framework
plt.figure(figsize=(10, 5))
plt.barh(names, min_times)
for i, v in enumerate(min_times):
    plt.text(v + 10, i, str(v))
plt.xlim([0, max(min_times) + 100])
plt.xlabel("Time (µs)")
plt.ylabel("Framework")
plt.tight_layout()
plt.savefig("data/min_time.png")
plt.show()
