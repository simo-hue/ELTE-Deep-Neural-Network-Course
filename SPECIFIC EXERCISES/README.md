# SPECIFIC EXERCISES — training for the DND coding test

Twelve notebooks that take you from "what shape is this tensor?" to a complete, timed mock test. They are built from this repository's material — the homework (tasks A–I), the lecturer's week 6–8 notebooks (Titanic, iterators, generators, `Dataset`/`DataLoader`), and the lab solutions (*Gradients and Regression*, *First Network*, *Tiers and Optimizers*) — but every exercise uses different data, so you cannot pass by recognising an old answer.

The coding test asks you to **train a network** for regression, classification, or both, in a bare Jupyter environment. The code for that is short. What actually decides the result is whether you can **predict what each line does** — shapes, dtypes, what a loss expects, what a curve means — because without tools that prediction is your only debugger. So the notebooks are organised around *understanding checks*, not around code to copy.

## How every notebook works

| Element | What to do |
|---|---|
| **Theory in 60 seconds** | A reminder, not a lesson. If something in it is unfamiliar, read the lecture slides for that topic first. |
| **Predict exercises** | Write down your prediction **before** running anything. A wrong prediction is the most valuable thing that can happen. |
| **Implement-by-hand exercises** | You rebuild what PyTorch does internally (the optimizer step, BCE, softmax, cross-entropy, weight decay, momentum…) and verify it against the library. After that, the library call is no longer a magic word. |
| **Bug hunts** | Broken code with no error message. You learn the *fingerprint* of each classic mistake. |
| `check("…", lambda: …)` | Prints ✅ / ❌. A ❌ with an exception in brackets usually means "not done yet". |
| ✍️ **Your answer** | Explain in your own words *before* opening the solution. If you cannot explain it, you have not understood it yet. |
| `▶ Hint` | Folded. Give yourself five minutes first. |
| **Blank-page drill** | The same task with **no scaffolding**, ideally in a new empty notebook and against the clock. This is the part that prepares you for the test. |

Solutions are in `SOLUTIONS/` — fully executed, so you can compare numbers and plots. (Your numbers may differ slightly from mine: random initialisation depends on the PyTorch version.) After reading a solution, close it and **retype** the answer from memory.

## The path

| # | Notebook | What you can do afterwards | Time |
|---|---|---|---|
| 01 | Tensors, shapes, dtypes | Predict the shape and dtype of any expression; spot the `(n, 1)` vs `(n,)` broadcast | 1.5 h |
| 02 | Autograd and gradient descent | Explain and re-implement each of the four lines of a training step; reason about the learning rate | 2 h |
| 03 | Regression and feature scaling | Train linear / logistic regression with `nn`; explain *why* unscaled features diverge; convert back to real units | 2 h |
| 04 | Data pipeline | Text → NaN handling → shuffle → split → fit-on-train preprocessing, vectorised; recognise leakage | 2 h |
| 05 | Batching and data loaders | Write slicing loops, generators, iterator classes, `Dataset`/`DataLoader`; know what shapes come out | 1.5 h |
| 06 | MLP regression, end to end | Homework D–F from an empty page: model class, train/val loop, curves, test loss | 2.5 h |
| 07 | Binary classification | Sigmoid+`BCELoss` vs logits+`BCEWithLogitsLoss`; accuracy, precision, recall, F1; weighted loss; balanced batches | 2.5 h |
| 08 | Multi-class classification | Softmax and cross-entropy by hand; label dtype/shape rules; confusion matrix; per-class metrics | 2.5 h |
| 09 | Training control | Early stopping with weight restoration; L2/L1; dropout and `train()`/`eval()`; SGD, momentum, Adam by hand | 2.5 h |
| 10 | Regression + classification together | Two-headed network, combined loss, loss scales, reporting in real units | 2 h |
| 11 | **Mock test A** | Regression, then 3-class classification — the homework's structure, new data, with a tester | 2.5 h |
| 12 | **Mock test B** | Binary + imbalance + multi-task, with every detail changed (csv module, min-max, own generator, sigmoid/BCELoss) | 2.5 h |

Do them in order: each one uses the previous ones, and 06–10 open with a repetition of earlier steps *from memory* on purpose.

## A possible schedule (coding test probably on 3 December)

- **Week 1 (to ~20 Nov):** 01 → 05, one per day. The same evening: that notebook's blank-page drill.
- **Week 2 (to ~27 Nov):** 06 → 10, one per day. Each morning, redo yesterday's blank-page drill in a new notebook before starting the next topic.
- **Last days:** mock A; two days later mock B; then repeat whichever went worse **from an empty copy**. The day before the test: only the three "cards" below, no new material.

## Make practice look like the test

- Use plain **Jupyter Lab**, the environment of the test. Switch autocomplete off (do not press Tab) — you will not have an IDE.
- Train yourself to use the **offline documentation**, which is allowed in the test: `help(nn.CrossEntropyLoss)`, `torch.optim.Adam?`, `np.nanmedian?`. Reading a docstring quickly is a skill; practise it now.
- After every cell, **verify without a tester**: print `.shape`, `.dtype`, `np.isnan(x).sum()`, the first rows, the class counts. In the test, these prints are your tester.
- Re-create the model and the optimizer inside the training cell, and keep `torch.manual_seed(...)` at its top, so a re-run starts fresh.

## Card 1 — the decision table

| | Regression | Binary classification | Multi-class classification |
|---|---|---|---|
| last layer | `Linear(h, 1)` | `Linear(h, 1)` (+ `Sigmoid` if `BCELoss`) | `Linear(h, k)`, **no** softmax |
| loss | `MSELoss` | `BCEWithLogitsLoss` — or `Sigmoid` + `BCELoss` | `CrossEntropyLoss` |
| label shape | `(batch, 1)` | `(batch, 1)` | `(batch,)` |
| label dtype | `float32` | `float32` | `int64` |
| prediction | the output | `logit > 0` / `prob > 0.5` | `logits.argmax(dim=1)` |
| untrained loss | ≈ mean(y²) | ≈ ln 2 = 0.69 | ≈ ln k |
| baseline | predict the training mean | majority class | majority class |
| metric | MSE / RMSE in real units | accuracy, precision, recall, F1 | accuracy, confusion matrix, per-class recall |

You should be able to *reconstruct* this table by reasoning (why must the regression label be `(batch, 1)`? why does cross-entropy want an index?) — notebooks 01, 07 and 08 give you the reasons.

## Card 2 — the order of a pipeline

parse → flags for missing values → **shuffle → split** → fit fill values and scaling **on the training split** → apply to all splits → separate inputs from labels → iterators → model → loop (train / validate each epoch) → curves → restore best weights → test metrics once.

## Card 3 — fingerprints of the silent bugs

| What you see | Most likely cause | Notebook |
|---|---|---|
| loss is `nan` from the first epoch | NaNs left in the data, or unscaled features with a large learning rate | 03, 04 |
| regression loss stuck near the variance of the target, all predictions nearly equal | label shape `(batch,)` against output `(batch, 1)` — or inputs and labels shuffled separately | 01, 06 |
| loss improves in epoch 0 and is then perfectly flat | generator created once outside the epoch loop; or `zero_grad()` between `backward()` and `step()` | 02, 05 |
| curve starts low and flat on a re-run | model not re-created in the training cell | 06 |
| loss explodes or oscillates | learning rate too high; `zero_grad()` forgotten | 02, 03 |
| binary loss cannot get below ~0.55–0.69, everything predicted as class 1 | `Sigmoid` in the model **and** `BCEWithLogitsLoss` | 07 |
| multi-class loss cannot get below ~0.74 (k = 4), trains slowly | `Softmax` in the model **and** `CrossEntropyLoss` | 08 |
| `RuntimeError: mat1 and mat2 must have the same dtype` | `float64` inputs (from NumPy) into `float32` layers | 01 |
| `expected scalar type Long but found Float` | class labels are not `int64` | 08 |
| validation much worse than training from the very first epoch | no shuffling before the split (file is sorted) | 04 |
| "restored" model is no better than the last epoch | `best_state = model.state_dict()` without `copy.deepcopy` | 09 |
| metrics change every time you evaluate | `model.eval()` forgotten (dropout active) | 09 |
| weighted loss has no effect | weights of shape `(batch,)` multiplied with losses of shape `(batch, 1)` | 07 |

## Files

```
SPECIFIC EXERCISES/
├── README.md                 this file
├── dnd_utils.py              check(), ready-made data loaders, the mock-test testers
├── data/                     houses.csv, rice.csv, bikes.csv, machines.csv (synthetic), titanic.csv (course dataset)
├── 01_… 12_….ipynb           the exercises
└── SOLUTIONS/                the same notebooks, solved and executed
```

The four synthetic datasets were generated for these exercises (with missing values, text categories and files sorted by a column on purpose). `dnd_utils.py` only gives feedback and data — nothing in it is needed, or available, in the real test.

Requirements: the course environment (`numpy`, `torch`, `matplotlib`, Jupyter). Start Jupyter from this folder or open the notebooks in place, so that `dnd_utils.py` and `data/` are found.
