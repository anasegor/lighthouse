import json
import matplotlib.pyplot as plt

with open("base_range_metrics.jsonl", "r") as f:
    data = json.load(f)

ranges = [k for k in data["brief"].keys() if "full" not in k and "R1@0.7" in k]
ranges_sorted = sorted(ranges, key=lambda x: x)

r1_values = []
range_labels = []
for r in ranges_sorted:
    if r.startswith("MR-"):
        range_name = r.split("-")[1]
        range_labels.append(range_name)
        r1_values.append(data["brief"][r])

# Строим гистограмму
plt.figure(figsize=(10, 6))
plt.bar(range_labels, r1_values, color='skyblue')
plt.xlabel("Диапазон длительности (сек)")
plt.ylabel("MR-R1@0.7 (%)")
plt.title("Гистограмма MR-R1@0.7 по диапазонам длительности")
plt.ylim(0, 50)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()