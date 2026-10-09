import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Generate Baseline Normal Data (2 key features for 2D visualization)
np.random.seed(42)
baseline_data = np.random.multivariate_normal(
    mean=[10.0, 50.0], 
    cov=[[1.2, 0.4], [0.4, 2.5]], 
    size=300
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(baseline_data)

detector = IsolationForest(contamination=0.05, random_state=42)
detector.fit(X_scaled)

# Incoming test samples (In-Distribution vs OOD Anomalies)
incoming_samples = np.array([
    [10.2, 49.8],
    [9.5, 51.2],
    [15.5, 62.0],  # Process Drift / Out-of-Distribution
    [6.0, 38.0]    # Material Defect / Out-of-Distribution
])

# Create Mesh Grid for Decision Boundary Graphic
x_min, x_max = baseline_data[:, 0].min() - 3, baseline_data[:, 0].max() + 5
y_min, y_max = baseline_data[:, 1].min() - 5, baseline_data[:, 1].max() + 15
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
Z = detector.decision_function(grid_scaled)
Z = Z.reshape(xx.shape)

# --- GRAPHICAL VISUALIZATION ---
plt.figure(figsize=(10, 6))

# Contour Map of Anomaly Score
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r, alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')  # OOD Threshold boundary

# Plot Training and Test Data Points
plt.scatter(baseline_data[:, 0], baseline_data[:, 1], c='gray', s=25, alpha=0.6, label='Baseline In-Spec Data')

test_preds = detector.predict(scaler.transform(incoming_samples))
for idx, (sample, pred) in enumerate(zip(incoming_samples, test_preds)):
    label = 'In-Distribution' if pred == 1 else 'REJECT: Out-of-Distribution'
    color = 'green' if pred == 1 else 'darkred'
    marker = 'o' if pred == 1 else 'X'
    plt.scatter(sample[0], sample[1], color=color, s=120, marker=marker, 
                edgecolor='black', zorder=10, label=f'Test Sample {idx+1} ({label})' if idx in [0, 2] else "")

plt.title("Production Out-of-Distribution (OOD) Detection Boundary", fontsize=12, fontweight='bold')
plt.xlabel("Sensor Feature 1 (Plasma Peak Amplitude)")
plt.ylabel("Sensor Feature 2 (Resonance Frequency Shift)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()