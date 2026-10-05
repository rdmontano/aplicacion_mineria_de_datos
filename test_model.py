import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parent / "app.py"
spec = importlib.util.spec_from_file_location("app_mineria", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert len(mod.raw_df) == 1520
assert len(mod.df) == 1500
assert round(mod.metricas_log["AUC"], 3) == 0.742
assert round(mod.metricas_log["Recall"], 3) == 0.671
assert mod.cm.tolist() == [[159, 68], [24, 49]]
assert round(mod.silhouette_k[2], 3) == 0.172
print("OK: datos, modelo, matriz de confusión y K-Means reproducen el informe.")
