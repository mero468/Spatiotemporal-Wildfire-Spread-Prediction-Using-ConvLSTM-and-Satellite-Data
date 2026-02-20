# ------------------------------------------------------------------
#  ConvLSTM Training – Manavgat Fire 2021 (5% blanks, 11× fire)
# ------------------------------------------------------------------

import numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, callbacks

tf.random.set_seed(42)

# ───────────── 1 | load ------------------------------------------------
P = Path("./data/patches")
X_tr, y_tr = np.load(P/"X_train.npy"), np.load(P/"y_train.npy")
X_va, y_va = np.load(P/"X_val.npy"),   np.load(P/"y_val.npy")
X_te, y_te = np.load(P/"X_test.npy"),  np.load(P/"y_test.npy")
print("📂 Train:", X_tr.shape, y_tr.shape)

# ───────────── 2 | tf.data + aug --------------------------------------
def tf_aug(x, y):
    if tf.random.uniform([]) < .5: x, y = tf.reverse(x, [3]), tf.reverse(y, [2])  # flip L/R
    if tf.random.uniform([]) < .5: x, y = tf.reverse(x, [2]), tf.reverse(y, [1])  # flip U/D
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)  # rotate 0-270°
    x = tf.map_fn(lambda fr: tf.image.rot90(fr, k), x)
    y = tf.image.rot90(y, k)
    return x, y

BATCH = 8
train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            .shuffle(1024, seed=42).batch(BATCH).map(tf_aug).prefetch(2))
val_ds   = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(BATCH).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((X_te, y_te)).batch(BATCH)

# ───────────── 3 | model ----------------------------------------------
model = models.Sequential([
    layers.Input(shape=(7, 32, 32, 3)),
    layers.ConvLSTM2D(32, (3,3), padding="same", activation="relu", return_sequences=True),
    layers.SpatialDropout3D(0.2),
    layers.ConvLSTM2D(32, (3,3), padding="same", activation="relu"),
    layers.Conv2D(1, (1,1), activation="sigmoid", padding="same")
])

# ───────────── 4 | loss: weighted BCE -------------------------------
pix_ratio = float(np.mean(y_tr)) or 1e-6  # e.g., 0.00266
pos_w     = min(120., 1.0 / pix_ratio)
print(f"⚖️  fire-pixel ratio = {pix_ratio:.4%} → POS_W = {pos_w:.1f}")

def weighted_bce(y_true, y_pred, eps=1e-7):
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    w = 1. + (pos_w - 1.) * y_true
    bce = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))
    return tf.reduce_mean(w * bce)

opt = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0)

model.compile(
    optimizer=opt,
    loss=weighted_bce,
    metrics=[
        tf.keras.metrics.Precision(thresholds=0.1, name="prec"),
        tf.keras.metrics.Recall(thresholds=0.1,    name="rec"),
        tf.keras.metrics.BinaryIoU(threshold=0.1,  name="iou")
    ])

# ───────────── 5 | callbacks ------------------------------------------
class Peek(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pred = (self.model.predict(X_te[:1])[0,...,0] > 0.1).astype(int)
        print(f"👁️  Epoch {epoch+1} – predicted fire pixels (1 sample): {pred.sum()}")

ck = callbacks.ModelCheckpoint("cvlstm_blanks.h5", monitor="val_iou",
                               save_best_only=True, mode="max", verbose=1)
es = callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                             monitor="val_iou", mode="max", verbose=1)

# ───────────── 6 | train ----------------------------------------------
# ───────────── 6 | train ─────────────────────────────────────────────
history = model.fit(                      # ← ① capture History
    train_ds, epochs=50, validation_data=val_ds,
    callbacks=[ck, es, Peek()]
)

# ───────────── 7 | final test ───────────────────────────────────────
print("\n🧪 Final evaluation on test set:")
model.evaluate(test_ds)

# ───────────── 8 | plot learning curves ────────────────────────────
import matplotlib.pyplot as plt
epochs = range(1, len(history.history["iou"]) + 1)

plt.figure(figsize=(6,3))
plt.plot(epochs, history.history["iou"],  label="Train IoU")
plt.plot(epochs, history.history["val_iou"], label="Val IoU")
best_ep = np.argmax(history.history["val_iou"]) + 1
plt.scatter(best_ep, history.history["val_iou"][best_ep-1],
            c="red", label=f"Early stop @ {best_ep}")
plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.title("Training vs Validation IoU")
plt.legend(); plt.tight_layout()
plt.savefig("learning_curve.png", dpi=300)     # ← ② PNG for your slide
plt.close()

