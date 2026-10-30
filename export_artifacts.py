# ================================================================
# PASTE THIS AS A NEW CELL AT THE VERY END OF YOUR COLAB NOTEBOOK
# (run it after the cell that creates `future_output`)
# It saves everything the Streamlit app needs into ./artifacts
# ================================================================
import os, json
os.makedirs('artifacts', exist_ok=True)

# 1) trained model
xgb.save_model('artifacts/xgb_model.json')

# 2) store table (raw, unfilled) -- the app rebuilds features from this
store.to_csv('artifacts/store.csv', index=False)

# 3) last 90 days of ACTUAL sales per sampled store (for the history chart)
history = df[df['Date'] > df['Date'].max() - pd.Timedelta(days=90)][['Store', 'Date', 'Sales']]
history.to_csv('artifacts/history.csv', index=False)

# 4) metadata: features, encoders (as plain lists), scores, importances
meta = {
    'features': features,
    'encoder_classes': {c: list(map(str, e.classes_)) for c, e in encoders.items()},
    'trained_stores': sorted(int(s) for s in sampled_store_ids),
    'last_date': str(df['Date'].max().date()),
    'rmspe': {k: float(v) for k, v in rmspe_scores.items()},
    'r2': {k: float(v) for k, v in r2_scores.items()},
    'importance': dict(zip(features, map(float, xgb.feature_importances_))),
    'n_train_rows': int(len(train)),
    'n_test_rows': int(len(test)),
}
with open('artifacts/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

import xgboost, sys
print('xgboost version  :', xgboost.__version__)
print('python version   :', sys.version.split()[0])
print('Saved:', os.listdir('artifacts'))

# 5) download the folder to your laptop
!zip -r artifacts.zip artifacts
from google.colab import files
files.download('artifacts.zip')
