#  Preprocess Manavgat Wildfire Data for ConvLSTM
#  * Input: 7-day stacks (NDVI, LST, WIND) → 32×32 patches
#  * Output: Next-day FIRE masks
#  * Fire patches duplicated ×10; 5% of blank patches retained
# ------------------------------------------------------------------

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ----------------------------- config -----------------------------
DATA_DIR   = Path('./data/raw')
OUT_DIR    = Path('./data/patches'); OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH      = 32
WINDOW     = 7
KEEP_BLANK_RATIO = 0.05      # Keep 5% of blank patches
DUP_FIRE         = 10        # Fire patches stored 11× total
VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SEED       = 42

np.random.seed(SEED)

# --------------------------- helpers ------------------------------
def load_stack(p):  # → (T, H, W)
    with rasterio.open(p) as src:
        return src.read(out_dtype='float32')

def upsample(arr, tgt_len):
    if arr.shape[0] >= tgt_len: return arr
    reps = int(np.ceil(tgt_len / arr.shape[0]))
    return np.tile(arr, (reps, 1, 1))[:tgt_len]

def n_ndvi(x):  return np.clip(x, 0, 1)
def n_lst(x):   return (np.clip(x, -40, 60) + 40) / 100.
def n_wind(x):  return np.clip(x, 0, 30) / 30.

def windows(arr, win):
    for t in range(arr.shape[0] - win):
        yield t, t + win

def patch_coords(arr, patch):
    h, w = arr.shape[-2:]
    for i in range(0, h - patch + 1, patch):
        for j in range(0, w - patch + 1, patch):
            yield i, j

# ------------------------ load & align ----------------------------
ndvi  = n_ndvi (load_stack(DATA_DIR / 'NDVI_500m.tif'))
lst   = n_lst  (load_stack(DATA_DIR / 'LST_500m.tif'))
wind  = n_wind (load_stack(DATA_DIR / 'WIND_500m.tif'))
fire  =         load_stack(DATA_DIR / 'FIRE_MANAVGAT_DAILY_500m.tif')

T = fire.shape[0]
ndvi, lst, wind = [upsample(a, T) for a in (ndvi, lst, wind)]

H = min(a.shape[1] for a in (ndvi, lst, wind, fire))
W = min(a.shape[2] for a in (ndvi, lst, wind, fire))
ndvi, lst, wind, fire = [a[:, :H, :W] for a in (ndvi, lst, wind, fire)]

print('✅ Aligned shapes:', ndvi.shape, lst.shape, wind.shape, fire.shape)

# ------------------------ patch sampling --------------------------
X_list, y_list = [], []
n_fire, n_blank = 0, 0

for t0, t1 in tqdm(list(windows(ndvi, WINDOW)), desc='Creating windows'):
    x_win  = np.stack([ndvi[t0:t1], lst[t0:t1], wind[t0:t1]], axis=-1)
    label2d = fire[t1]

    for i, j in patch_coords(label2d, PATCH):
        y_patch = label2d[i:i+PATCH, j:j+PATCH]
        x_patch = x_win[:, i:i+PATCH, j:j+PATCH, :]
        x_patch = np.nan_to_num(x_patch, nan=0., posinf=0., neginf=0.)

        if y_patch.max():  # fire patch
            for _ in range(DUP_FIRE + 1):
                X_list.append(x_patch.astype('float32'))
                y_list.append(y_patch.astype('float32'))
            n_fire += 1
        else:  # blank patch
            if np.random.rand() < KEEP_BLANK_RATIO:
                X_list.append(x_patch.astype('float32'))
                y_list.append(y_patch.astype('float32'))
                n_blank += 1

print(f'🔥 Fire patches (×{DUP_FIRE+1}): {n_fire * (DUP_FIRE+1)}')
print(f'🟦 Blank patches kept: {n_blank}')
print(f'📦 Patch class ratio (fire to total): '
      f'{(n_fire*(DUP_FIRE+1))/(n_fire*(DUP_FIRE+1)+n_blank):.2f}')

X = np.stack(X_list)
y = np.stack(y_list)[..., None]  # add channel

print(f'✅ Dataset built: X {X.shape}   y {y.shape}   fire-pixel ratio={y.mean():.3%}')

# ---------------------- save train/val/test -----------------------
X_t, X_te, y_t, y_te = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=SEED,
    stratify=y.max((1,2,3)))

rel_val = VAL_RATIO / (1 - TEST_RATIO)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_t, y_t, test_size=rel_val, random_state=SEED,
    stratify=y_t.max((1,2,3)))

for name, array in [('X_train', X_tr), ('y_train', y_tr),
                    ('X_val', X_val), ('y_val', y_val),
                    ('X_test', X_te), ('y_test', y_te)]:
    np.save(OUT_DIR / f'{name}.npy', array)

print('✅ All splits saved to:', OUT_DIR.resolve())
