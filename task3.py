import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create 100 synthetic delivery records, including three long deliveries.
np.random.seed(42)
regular_times = np.clip(np.random.normal(32, 6, size=97), 15, 48)
delivery_times = np.concatenate([np.round(regular_times, 1), [68, 71, 75]])
pd.DataFrame({
    "OrderID": np.arange(1, 101),
    "DeliveryTime(min)": delivery_times,
}).to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")
real_times = df["DeliveryTime(min)"]
mean = real_times.mean()
std = real_times.std()
print(f"Real delivery times -- Mean: {mean:.1f} min Std: {std:.1f} min")

# Simulate expected times using the real feature's mean and standard deviation.
simulated_times = np.random.normal(mean, std, size=100)
print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")

# Use identical bins for both distributions to make comparison easier.
bins = np.histogram_bin_edges(
    np.concatenate([real_times.dropna().to_numpy(), simulated_times]), bins=15
)
plt.figure()
plt.hist(real_times.dropna(), bins=bins, alpha=0.6,
         label="Real (synthetic) deliveries", edgecolor="black")
plt.hist(simulated_times, bins=bins, alpha=0.6,
         label="Simulated expected deliveries", edgecolor="black")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Number of Deliveries")
plt.title("Real vs. Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

print("\nMissing values per column:")
print(df.isna().sum())

plt.figure()
plt.boxplot(real_times.dropna())
plt.xticks([1], ["Real Deliveries"])
plt.ylabel("Delivery Time (min)")
plt.title("Boxplot of Real Delivery Times")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

q1, median, q3 = real_times.quantile([0.25, 0.50, 0.75])
print(f"\nPercentiles -- 25th: {q1:.1f} 50th: {median:.1f} 75th: {q3:.1f}")
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"IQR outlier bounds: {lower_bound:.1f} - {upper_bound:.1f}")

# Flag unusually long deliveries using the upper IQR bound.
df["UnusuallyLong"] = real_times > upper_bound
long_deliveries = df[df["UnusuallyLong"]]
print(f"Unusually long outliers detected: {len(long_deliveries)} deliveries")
print(long_deliveries.to_string(index=False))
percentage = len(long_deliveries) / len(df) * 100
print(f"Percentage flagged as outliers: {percentage:.1f}%")
print("\nSaved delivery_comparison_histogram.png, delivery_boxplot.png")
