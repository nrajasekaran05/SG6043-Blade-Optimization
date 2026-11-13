# %%
# TRAIN_MODEL_CPU.PY - Stable CPU version
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers

print("="*60)
print("NEURAL NETWORK TRAINING (CPU)")
print("="*60)

# STEP 1: LOAD DATA
print("\nStep 1: Loading data...")

data_dir = DATA_DIR
results_file = data_dir + 'aerosandbox_results.csv'

df = pd.read_csv(results_file)
print(f"✓ Loaded {len(df)} data points")

df = df.dropna()
df = df[np.isfinite(df['cl']) & np.isfinite(df['cd'])]
print(f"✓ After cleaning: {len(df)} data points")

print(f"\nDataset statistics:")
print(f"  Unique designs: {df['design_id'].nunique()}")
print(f"  Alpha range: [{df['alpha'].min():.1f}°, {df['alpha'].max():.1f}°]")
print(f"  Cl range: [{df['cl'].min():.3f}, {df['cl'].max():.3f}]")
print(f"  Cd range: [{df['cd'].min():.5f}, {df['cd'].max():.5f}]")

# STEP 2: PREPARE FEATURES
print("\nStep 2: Preparing features and targets...")

feature_cols = [f'upper_c{i}' for i in range(9)] + \
               [f'lower_c{i}' for i in range(9)] + \
               ['alpha']

X = df[feature_cols].values
y = df[['cl', 'cd', 'cm']].values

print(f"✓ Features shape: {X.shape}")
print(f"✓ Targets shape: {y.shape}")

# STEP 3: SPLIT DATA
print("\nStep 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# STEP 4: NORMALIZE
print("\nStep 4: Normalizing data...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print("✓ Data normalized")

import joblib
joblib.dump(scaler_X, data_dir + 'scaler_X.pkl')
joblib.dump(scaler_y, data_dir + 'scaler_y.pkl')
print("✓ Scalers saved")

# STEP 5: BUILD MODEL
print("\nStep 5: Building neural network...")

model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("✓ Model built")
model.summary()

# STEP 6: TRAIN
print("\nStep 6: Training model (CPU mode)...")
print("This will take ~5-10 minutes...\n")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✓ Training complete!")

# STEP 7: EVALUATE
print("\nStep 7: Evaluating model...")

test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"✓ Test Loss (MSE): {test_loss:.6f}")
print(f"✓ Test MAE: {test_mae:.6f}")

y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

from sklearn.metrics import r2_score

r2_cl = r2_score(y_test[:, 0], y_pred[:, 0])
r2_cd = r2_score(y_test[:, 1], y_pred[:, 1])
r2_cm = r2_score(y_test[:, 2], y_pred[:, 2])

print(f"\nR² Scores:")
print(f"  Cl: {r2_cl:.4f}")
print(f"  Cd: {r2_cd:.4f}")
print(f"  Cm: {r2_cm:.4f}")
print(f"  Average: {np.mean([r2_cl, r2_cd, r2_cm]):.4f}")

# STEP 8: SAVE
print("\nStep 8: Saving model...")

model.save(data_dir + 'airfoil_surrogate_model.keras')
print(f"✓ Model saved")

# STEP 9: VISUALIZE
print("\nStep 9: Creating visualizations...")

fig = plt.figure(figsize=(15, 10))

# Training history
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Train', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Cl prediction
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=20)
ax2.plot([y_test[:, 0].min(), y_test[:, 0].max()],
         [y_test[:, 0].min(), y_test[:, 0].max()],
         'r--', linewidth=2)
ax2.set_xlabel('Actual Cl')
ax2.set_ylabel('Predicted Cl')
ax2.set_title(f'Cl Prediction (R²={r2_cl:.3f})')
ax2.grid(True, alpha=0.3)

# Cd prediction
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, s=20)
ax3.plot([y_test[:, 1].min(), y_test[:, 1].max()],
         [y_test[:, 1].min(), y_test[:, 1].max()],
         'r--', linewidth=2)
ax3.set_xlabel('Actual Cd')
ax3.set_ylabel('Predicted Cd')
ax3.set_title(f'Cd Prediction (R²={r2_cd:.3f})')
ax3.grid(True, alpha=0.3)

# Cl error
ax4 = plt.subplot(2, 3, 4)
cl_error = y_pred[:, 0] - y_test[:, 0]
ax4.hist(cl_error, bins=30, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Cl Prediction Error')
ax4.set_ylabel('Frequency')
ax4.set_title('Cl Error Distribution')
ax4.grid(True, alpha=0.3)

# Cd error
ax5 = plt.subplot(2, 3, 5)
cd_error = y_pred[:, 1] - y_test[:, 1]
ax5.hist(cd_error, bins=30, edgecolor='black', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Cd Prediction Error')
ax5.set_ylabel('Frequency')
ax5.set_title('Cd Error Distribution')
ax5.grid(True, alpha=0.3)

# Cm prediction
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_test[:, 2], y_pred[:, 2], alpha=0.5, s=20)
ax6.plot([y_test[:, 2].min(), y_test[:, 2].max()],
         [y_test[:, 2].min(), y_test[:, 2].max()],
         'r--', linewidth=2)
ax6.set_xlabel('Actual Cm')
ax6.set_ylabel('Predicted Cm')
ax6.set_title(f'Cm Prediction (R²={r2_cm:.3f})')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(data_dir + 'training_results.png', dpi=150)
plt.show()

print(f"✓ Visualization saved")

# SUMMARY
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nPerformance:")
print(f"  Test MAE: {test_mae:.6f}")
print(f"  R² Cl: {r2_cl:.4f}")
print(f"  R² Cd: {r2_cd:.4f}")
print(f"  R² Cm: {r2_cm:.4f}")
print(f"  Average R²: {np.mean([r2_cl, r2_cd, r2_cm]):.4f}")

print(f"\n✅ Model saved: {data_dir}airfoil_surrogate_model.keras")
print("="*60)