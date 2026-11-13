"""Small helpers for the SPECIFIC EXERCISES notebooks.

Nothing here is needed in the coding test -- it only gives you feedback (`check`)
and ready-made data (`load_*`) so each notebook can focus on one topic.
"""
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def check(name, cond):
    """Print a green/red line. `cond` is a zero-argument function returning True/False.

    Wrapped in a lambda so an unfinished exercise prints a red line instead of crashing:
        check("y has shape (n, 1)", lambda: y.shape == (n, 1))
    """
    try:
        ok = bool(cond())
    except Exception as e:                       # unfinished / wrong type / wrong shape ...
        print(f"❌ {name}   ({type(e).__name__}: {e})")
        return False
    print(("✅ " if ok else "❌ ") + name)
    return ok


def _read(name, delimiter):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _num(records, col):
    return np.array([float(r[col]) if r[col] != "" else np.nan for r in records], dtype=np.float32)


def _onehot(records, col, levels):
    v = np.array([r[col] for r in records])
    return (v[:, None] == np.array(levels)[None, :]).astype(np.float32)


def _finish(Xnum, Xbin, targets, seed, ratios=(0.6, 0.2)):
    """shuffle -> split -> impute (train median) + flags -> standardise numeric columns (train stats)."""
    n = len(Xnum)
    perm = np.random.default_rng(seed).permutation(n)
    Xnum, Xbin = Xnum[perm], Xbin[perm]
    targets = {k: v[perm] for k, v in targets.items()}
    a, b = int(n * ratios[0]), int(n * ratios[0]) + int(n * ratios[1])
    flags = np.isnan(Xnum).astype(np.float32)
    flags = flags[:, flags.any(axis=0)]                       # one flag per column that has gaps
    med = np.nanmedian(Xnum[:a], axis=0)
    Xnum = np.where(np.isnan(Xnum), med, Xnum)
    mean, std = Xnum[:a].mean(axis=0), Xnum[:a].std(axis=0)
    X = np.concatenate([(Xnum - mean) / std, Xbin, flags], axis=1).astype(np.float32)
    out = {}
    for name, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
        out["X_" + name] = X[sl]
        for k, v in targets.items():
            out[f"{k}_{name}"] = v[sl]
    return out


def load_houses(seed=0):
    """houses.csv, fully preprocessed. Returns a dict of float32 NumPy arrays:

    X_train/X_val/X_test : (n, 13)  4 standardised numeric cols | garden | 5 district one-hot |
                           district-missing flag | age-missing flag | dist-missing flag
                           (rooms is treated as numeric: area, rooms, age, dist)
    y_*                  : (n,)     price in thousand EUR   (regression target)
    s_*                  : (n,)     sold_fast, 0.0 / 1.0     (binary target)
    """
    r = _read("houses.csv", ",")
    Xnum = np.stack([_num(r, c) for c in ("area_m2", "rooms", "age_years", "dist_center_km")], axis=1)
    Xbin = np.concatenate([_num(r, "garden")[:, None],
                           _onehot(r, "district", ["Buda", "Pest", "Obuda", "Ujpest", "Csepel"])], axis=1)
    flags_district = (np.array([x["district"] for x in r]) == "").astype(np.float32)[:, None]
    Xbin = np.concatenate([Xbin, flags_district], axis=1)
    return _finish(Xnum, Xbin, {"y": _num(r, "price_keur"), "s": _num(r, "sold_fast")}, seed)


def load_titanic_raw():
    """titanic.csv as the lecturer's notebook loads it.

    inputs : (891, 7) float32  Pclass, Sex(0 female/1 male), Age, SibSp, Parch, Fare, Embarked(0 C/1 Q/2 S), NaN = missing
    labels : (891,)   int32    Survived
    """
    r = _read("titanic.csv", ",")
    sex = {"female": 0.0, "male": 1.0}
    emb = {"C": 0.0, "Q": 1.0, "S": 2.0, "": np.nan}
    cols = [_num(r, "Pclass"), np.array([sex[x["Sex"]] for x in r], dtype=np.float32), _num(r, "Age"),
            _num(r, "SibSp"), _num(r, "Parch"), _num(r, "Fare"),
            np.array([emb[x["Embarked"]] for x in r], dtype=np.float32)]
    return np.stack(cols, axis=1), np.array([int(x["Survived"]) for x in r], dtype=np.int32)


def load_titanic(seed=0):
    """titanic.csv fully preprocessed: X_* (n, 10) float32, y_* (n,) float32 in {0, 1}.
    Columns: Pclass, Age, SibSp, Parch, Fare (standardised) | Sex | Embarked one-hot C,Q,S | Age-missing flag."""
    inputs, labels = load_titanic_raw()
    Xnum = inputs[:, [0, 2, 3, 4, 5]]
    e = inputs[:, 6]
    Xbin = np.concatenate([inputs[:, [1]], (e[:, None] == np.arange(3)[None, :]).astype(np.float32)], axis=1)
    return _finish(Xnum, Xbin, {"y": labels.astype(np.float32)}, seed)


RICE_CLASSES = ["arborio", "basmati", "jasmine", "carnaroli"]


def load_rice(seed=0):
    """rice.csv fully preprocessed: X_* (n, 9) float32, y_* (n,) int64 in {0,1,2,3} (index into RICE_CLASSES).
    Columns: 4 standardised numeric | colour one-hot white,brown,red | 2 missing-flags (weight, moisture)."""
    r = _read("rice.csv", ";")
    Xnum = np.stack([_num(r, c) for c in ("length_mm", "width_mm", "weight_mg", "moisture_pct")], axis=1)
    Xbin = _onehot(r, "color", ["white", "brown", "red"])
    y = np.array([RICE_CLASSES.index(x["variety"]) for x in r], dtype=np.int64)
    return _finish(Xnum, Xbin, {"y": y}, seed)


def reference_04(seed=0):
    """Reference results for notebook 04 (used only by its check cells)."""
    r = _read("houses.csv", ",")
    D = ["Buda", "Pest", "Obuda", "Ujpest", "Csepel"]
    didx = np.array([D.index(x["district"]) if x["district"] else np.nan for x in r], dtype=np.float32)
    raw = np.stack([_num(r, "area_m2"), _num(r, "rooms"), _num(r, "age_years"), _num(r, "dist_center_km"),
                    didx, _num(r, "garden"), _num(r, "price_keur"), _num(r, "sold_fast")], axis=1)
    data = np.concatenate([raw, (~np.isnan(raw[:, [2, 3, 4]])).astype(np.float32)], axis=1)
    sh = data[np.random.default_rng(seed).permutation(len(data))]
    a, b = int(len(sh) * 0.6), int(len(sh) * 0.6) + int(len(sh) * 0.2)
    tr, va, te = sh[:a], sh[a:b], sh[b:]
    med = np.nanmedian(tr[:, [2, 3]], axis=0)

    def build(s):
        num = s[:, :4].copy()
        num[:, 2] = np.where(np.isnan(num[:, 2]), med[0], num[:, 2])
        num[:, 3] = np.where(np.isnan(num[:, 3]), med[1], num[:, 3])
        oh = (s[:, 4, None] == np.arange(5)[None, :]).astype(np.float32)
        return np.concatenate([num, s[:, 5, None], oh, s[:, 8:11]], axis=1)

    Xtr, Xva, Xte = build(tr), build(va), build(te)
    mean, std = Xtr[:, :4].mean(axis=0), Xtr[:, :4].std(axis=0)

    def scale(X):
        X = X.copy(); X[:, :4] = (X[:, :4] - mean) / std
        return X

    return dict(raw=raw, data=data, train=tr, val=va, test=te, med=med, X_train_unscaled=Xtr, mean=mean, std=std,
                X_train=scale(Xtr), X_val=scale(Xva), X_test=scale(Xte),
                y_train=tr[:, 6, None], y_val=va[:, 6, None], y_test=te[:, 6, None])


def reference_mock_a():
    """Reference results for the deterministic parts of mock test A (used only by its Tester)."""
    S = ["winter", "spring", "summer", "autumn"]
    r = _read("bikes.csv", ";")
    cols = ["hour", "temp_c", "humidity", "wind_kmh", "weekday", "season", "holiday", "rentals"]
    raw = np.stack([np.array([float(S.index(x[c])) for x in r], dtype=np.float32) if c == "season" else _num(r, c)
                    for c in cols], axis=1)
    sh = raw[np.random.default_rng(42).permutation(len(raw))]
    a, b = int(len(sh) * 0.6), int(len(sh) * 0.6) + int(len(sh) * 0.2)
    splits = [sh[:a], sh[a:b], sh[b:]]
    med = np.nanmedian(splits[0][:, 1:4], axis=0)

    def build(s):
        num = s[:, [0, 1, 2, 3, 4]].copy()
        num[:, 1:4] = np.where(np.isnan(num[:, 1:4]), med, num[:, 1:4])
        oh = (s[:, 5, None] == np.arange(4)[None, :]).astype(np.float32)
        return np.concatenate([num, s[:, 6, None], oh, (~np.isnan(s[:, 1:4])).astype(np.float32)], axis=1)

    Xs = [build(s) for s in splits]
    mean, std = Xs[0][:, :5].mean(axis=0), Xs[0][:, :5].std(axis=0)
    for X in Xs:
        X[:, :5] = (X[:, :5] - mean) / std
    ys = [s[:, 7] for s in splits]
    cls = [np.where(y < 25, 0, np.where(y < 100, 1, 2)).astype(np.int64) for y in ys]
    return dict(raw=raw, splits=splits, med=med, X=Xs, y=ys, cls=cls)


class TesterA:
    """Mimics the homework's Tester for mock test A: prints OK / FAILED per sub-task."""

    def __init__(self):
        self.ref = reference_mock_a()

    def get_dataset_content(self):
        return (DATA / "bikes.csv").read_text(encoding="utf-8")

    def _report(self, name, problems):
        print(f"Tester: {name} " + ("OK" if not problems else "FAILED -> " + "; ".join(problems)))

    def test(self, name, *args):
        import torch
        R, p = self.ref, []
        try:
            if name == "dataset_load":
                (a,) = args
                if getattr(a, "shape", None) != (2000, 8): p.append(f"shape should be (2000, 8), got {getattr(a, 'shape', None)}")
                elif a.dtype != np.float32: p.append(f"dtype should be float32, got {a.dtype}")
                elif not np.allclose(a, R["raw"], equal_nan=True): p.append("values (or NaN positions) differ")
            elif name == "dataset_split":
                for got, exp, nm in zip(args, R["splits"], ("train", "val", "test")):
                    if got.shape != exp.shape: p.append(f"{nm}: shape {got.shape}, expected {exp.shape}")
                    elif not np.allclose(got, exp, equal_nan=True): p.append(f"{nm}: rows differ (seed 42? shuffle before split?)")
            elif name == "preprocess":
                for got, exp, nm in zip(args, R["X"], ("X_train", "X_val", "X_test")):
                    if got.shape != exp.shape: p.append(f"{nm}: shape {got.shape}, expected {exp.shape}")
                    elif got.dtype != np.float32: p.append(f"{nm}: dtype {got.dtype}")
                    elif np.isnan(got).any(): p.append(f"{nm}: contains NaN")
                    elif not np.allclose(got, exp, atol=1e-4): p.append(f"{nm}: values differ (column order? train statistics?)")
            elif name in ("reg_iter", "cl_iter"):
                tr, va, te = args
                want_y = R["y"] if name == "reg_iter" else R["cls"]
                for dl, X, y, nm in zip((tr, va, te), R["X"], want_y, ("train", "val", "test")):
                    xb, yb = next(iter(dl))
                    if xb.dtype != torch.float32 or xb.shape[1:] != (13,): p.append(f"{nm}: inputs should be float32 (batch, 13), got {xb.dtype} {tuple(xb.shape)}")
                    if name == "reg_iter" and (yb.dtype != torch.float32 or yb.ndim != 2 or yb.shape[1] != 1): p.append(f"{nm}: labels should be float32 (batch, 1), got {yb.dtype} {tuple(yb.shape)}")
                    if name == "cl_iter" and (yb.dtype != torch.int64 or yb.ndim != 1): p.append(f"{nm}: labels should be int64 (batch,), got {yb.dtype} {tuple(yb.shape)}")
                    ally = torch.cat([b[1].reshape(-1) for b in dl]).numpy()
                    if len(ally) != len(y): p.append(f"{nm}: {len(ally)} samples, expected {len(y)}")
                    elif not np.allclose(np.sort(ally), np.sort(y)): p.append(f"{nm}: label values differ")
                    elif nm == "test" and not np.allclose(ally, y): p.append("test: order must not change")
                    elif nm != "test" and np.allclose(ally, y): p.append(f"{nm}: must be traversed in random order")
            elif name in ("reg_model_architecture", "cl_model_architecture"):
                (m,) = args
                want = [(64, 13), (32, 64), (1, 32)] if name.startswith("reg") else [(40, 13), (20, 40), (3, 20)]
                got = [tuple(l.weight.shape) for l in m.modules() if isinstance(l, torch.nn.Linear)]
                if got != want: p.append(f"Linear weight shapes {got}, expected {want}")
                if any(isinstance(l, (torch.nn.Softmax, torch.nn.Sigmoid, torch.nn.LogSoftmax)) for l in m.modules()): p.append("no Softmax/Sigmoid layer expected in the model")
                out = m(torch.zeros(4, 13))
                if tuple(out.shape) != (4, want[-1][0]): p.append(f"output shape {tuple(out.shape)}")
            elif name == "reg_model_learning":
                (mse,) = args
                base = float(((R["y"][2] - R["y"][0].mean()) ** 2).mean())
                if not (mse < 0.3 * base): p.append(f"test MSE {mse:.0f} is not clearly below the baseline {base:.0f} (expected < {0.3 * base:.0f})")
                if mse < 100: p.append("suspiciously good: is the label among the inputs?")
            elif name == "cl_model_learning":
                ce, acc = args
                if not (0.25 < ce < 0.75): p.append(f"test CE {ce:.3f} outside the expected range 0.25-0.75")
                if not (0.72 < acc < 0.95): p.append(f"test accuracy {acc:.3f} outside the expected range 0.72-0.95")
            else:
                p.append("unknown test name")
        except Exception as e:
            p.append(f"{type(e).__name__}: {e}")
        self._report(name, p)
        return not p


def reference_mock_b():
    """Reference results for the deterministic parts of mock test B (used only by its check cells)."""
    r = _read("machines.csv", ",")
    T, SH = ["A", "B", "C"], ["day", "night"]
    raw = np.stack([_num(r, "temp_c"), _num(r, "vibration"), _num(r, "pressure"), _num(r, "rpm"),
                    _num(r, "hours_since_service"),
                    np.array([T.index(x["type"]) for x in r], dtype=np.float32),
                    np.array([SH.index(x["shift"]) for x in r], dtype=np.float32),
                    _num(r, "wear"), _num(r, "failure")], axis=1)
    sh = raw[np.random.default_rng(7).permutation(len(raw))]
    a, b = int(len(sh) * 0.7), int(len(sh) * 0.7) + int(len(sh) * 0.15)
    splits = [sh[:a], sh[a:b], sh[b:]]
    fill = np.nanmean(splits[0][:, [1, 2]], axis=0)

    def build(s):
        num = s[:, :5].copy()
        num[:, 1:3] = np.where(np.isnan(num[:, 1:3]), fill, num[:, 1:3])
        oh = (s[:, 5, None] == np.arange(3)[None, :]).astype(np.float32)
        return np.concatenate([num, s[:, 6, None], oh, np.isnan(s[:, 1:3]).astype(np.float32)], axis=1)

    Xs = [build(s) for s in splits]
    lo, hi = Xs[0][:, :5].min(axis=0), Xs[0][:, :5].max(axis=0)
    for X in Xs:
        X[:, :5] = (X[:, :5] - lo) / (hi - lo)
    return dict(raw=raw, splits=splits, fill=fill, X=Xs, wear=[s[:, 7] for s in splits], fail=[s[:, 8] for s in splits])
