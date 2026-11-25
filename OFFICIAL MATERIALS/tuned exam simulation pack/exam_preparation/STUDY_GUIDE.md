# Preparing for the ELTE Deep Network Development coding test

**Start with the 35-minute diagnostic.** Your existing notebooks already cover most of the underlying PyTorch. The best use of your next sessions is to build the pipeline from raw text, switch between binary and multiclass conventions without hints, and finish the requested analysis under time pressure.

[Diagnostic](00_diagnostic.ipynb) · [Binary mock](01_binary_mock.ipynb) · [Multiclass mock](02_multiclass_mock.ipynb) · [Worked solutions](solutions/)

## What the two released papers tell you

The autumn example pack contains two spring 2026 papers. Each gives 180 minutes and 30 points for a single tabular classification problem. Both supply the CSV contents as strings and ask you to derive labels from three experts’ scores. That is enough evidence to prioritize a particular workflow. The autumn questions, point split and dataset are still unknown.

| Graded work | A · 21 April · binary | B · 12 May · three classes |
|---|---|---|
| Read and prepare the tables | A, 1 pt. Parse two semicolon files as float32. B, 6 pts. Reorder the second table to the first header, then stack rows. Full credit needs both arbitrary permutations and vectorized reordering. | A, 1 pt. Parse one comma file as float32. B, 0/3/5 pts. Keep the table, or insert two validity flags, or compute both flags together. Zeros in the first two features indicate missingness. Preserve their original values. |
| Define labels and split | C, 3 pts. Premium if mean score > 5 and minimum ≥ 3. Remove all three score columns. D, 3 pts. Shuffle **within each category**, then split each 70/15/15 and print all six shapes. | C, 3 pts. Shuffle complete rows together, then allocate 1,497 train, 600 validation and 600 test. In D derive labels from scores. Bottom if minimum ≤ 2, top if minimum ≥ 5 and mean ≥ 6, otherwise middle. |
| Make batches | E, 3 pts. Mixed classes in all loaders. Inputs are float32 (B, 11), labels float32 (B, 1). Premium is 1. | D, 6 pts including labels. Print counts for all three classes in every split. Inputs are float32 (B, 11) or (B, 13) depending on the flags. Targets are int64 (B,). |
| Define the model | F, 2 pts. Subclass nn.Module and choose a suitable binary architecture. One output per sample. | E, 3 pts. Subclass nn.Module. Use two linear layers, 20 hidden units, ReLU, dropout 0.10 after ReLU, then three outputs. |
| Train and evaluate | G, 6 pts. Train with binary cross-entropy, validate each epoch, restore the best weights, plot both losses with a legend, report test BCE and accuracy, and print predicted/true examples. | F, 6 pts. Same training discipline, with multiclass cross-entropy. Also plot a confusion matrix with true rows, predicted columns, class ticks and counts in the cells. |
| Analyze results | H, 3 pts. Across the full test set find the five probabilities closest to 0.5 and print their inputs, probabilities and true labels. I, 3 pts. Histogram predicted probabilities separately by **true** class. | G, 3 + 3 pts. For each class, treat membership as a binary decision, count TP and TN, divide by N, then average. Base credit permits a class loop. Full credit requires vectorizing over classes too. Use Python, NumPy or PyTorch for G. |

**What to memorize.** The data flow and the two output conventions. For a binary model with logits, use one logit, BCEWithLogitsLoss and a zero threshold. Apply sigmoid once for probabilities. A model that already outputs a sigmoid probability can use BCELoss and a 0.5 threshold. For multiclass use three raw logits, CrossEntropyLoss and argmax over the class dimension. Check the prediction and target shapes before comparing them. A (B, 1) vs (B,) comparison can silently produce B × B decisions.

**What to read afresh.** Circle the boundary words, split unit, flag polarity, requested columns, architecture and metric definition. In B a valid sensor gets 1. In A a juice with ratings [3, 5, 7] has mean exactly 5 and is budget. The official A and B rules give different meanings to the same scores. Giving the network any score column lets it see the answer.

**The unusual metric.** Consider true labels [0, 2, 1, 1, 1] and predictions [0, 0, 2, 2, 1] from B. Ordinary accuracy is 2/5. The three one-vs-rest accuracies are 4/5, 3/5 and 2/5, averaging to 0.60. Mean per-class recall gives a different answer. For a three-class problem with exactly one true and predicted class per item, each wrong guess fails two of the three membership decisions. That gives a quick check: macro accuracy must equal 1 − (2/3)(1 − ordinary accuracy). Implement the requested per-class computation so you can show how you got it.

## Which existing exercises earn their study time?

I compared the prompts and the saved outputs in the student copies. A saved check records one run. The fresh diagnostic tells you what you can reproduce unaided.

| Existing notebooks | Evidence and next action |
|---|---|
| 01 · Shapes; 02 · Autograd; 03 · Regression and scaling | Their saved checks include successful shape, dtype, mask and gradient exercises. Keep 1.6–1.7 handy. Use 02/03 to debug a specific issue, then move on. Deriving optimizers and converting regression coefficients back to original units have little direct payoff on these two papers. |
| 04 · Data pipeline | Stepwise parsing, flags, splitting and scaling checks pass. The final six checks fail together because the single blank-page function still raises NotImplementedError. Redo the assembly once from raw text. For official B, follow the specified zero-and-flag representation. |
| 05 · Batching | Saved checks pass for mixed tensor shapes, re-iteration and the sample-weighted epoch mean in 5.8. Reuse the simplest DataLoader pattern. Keep inputs and labels together when shuffling. Check one batch and the number of items seen in a complete pass. |
| 06 · Regression MLP | Training, evaluation and loss-curve work transfers directly. Reuse the loop from 6.4–6.6 and swap the loss and prediction rule. A fresh end-to-end regression run is lower priority. |
| 07 · Binary classification | Saved checks pass for logits vs probabilities, accuracy and training. Transfer these to probabilities gathered over the full test set. Practise ranking and plotting them by true label. Weighted BCE and threshold tuning are lower priority here. |
| 08 · Multiclass; 09 · Training control | Highest priority. These student copies have no saved execution counts. Do 8.2 and 8.4–8.7, then 9.2, 9.3 and 9.5. Write out early stopping from memory once. Compare ordinary accuracy, mean recall and B’s macro accuracy on one small matrix. |
| 10 · Multitask; 11–12 · Old mocks | Good broader course practice. The old mocks use 150 minutes and add regression or multitask objectives. Use their classification portions if you need repetition. Use the released papers to judge readiness for this pack. |

## A sequence for someone who already codes

1. **First session, 35 minutes.** Take the diagnostic from a fresh kernel without looking at the solutions. If the label boundaries or metrics fail, explain a failing row by hand. If the checkpoint drill fails, revisit 9.2–9.3. Skip the review for any part you can already implement and explain.
2. **Second session, 60–90 minutes.** Finish the relevant parts of 08 and 09. Run one real validation pass. Save an independent best state, restore it, and verify that re-evaluation reproduces the minimum validation loss. Do 04’s final function if assembling the data pipeline still takes several attempts.
3. **First rehearsal.** Take the new mock that matches your weaker branch. Mark each task as unaided, completed with documentation, or incomplete. Award points for plots and printed examples only when the requested information is visible. Open its worked solution after scoring and redo the slowest cell from blank.
4. **Official rehearsal.** Make a working copy of the corresponding official notebook beside its tester and zh_files folder. Take the full 180 minutes with the allowed resources. If you already know that paper’s answers, use the fresh mock for the timed attempt and the official paper for checking exact requirements.
5. **Switch branches.** Use the other new mock for targeted practice, then sit the other official paper under the same conditions. Record where minutes or points were lost. Stop repeating a section when you can complete it on a different input and explain a boundary case. No need to wait for a specific calendar date.

### A 180-minute pacing rehearsal

| Elapsed minutes | Evidence to have on screen |
|---|---|
| 0–10 | Read the whole paper. Note the output shapes, split sizes, graded alternatives and final plots. |
| 10–55 | Parsed rows, matching headers or flag columns, label-rule spot checks and verified splits. |
| 55–80 | Three reusable loaders, one mixed batch with the correct dtype and a working forward pass. |
| 80–130 | Train and validation histories, automatic best-weight restoration, test loss and accuracy. |
| 130–160 | Required example predictions and every analysis plot or metric. Check each against the wording. |
| 160–180 | Save, restart and run all in order, review outputs and variable names, then prepare the requested notebook. |

If a bonus blocks the rest, take the viable base route and continue. A’s merge can start with a correct explicit ordering. B explicitly permits carrying the unmodified table forward for zero points on its flags task. Keep the model input width tied to the actual feature count so an improved preprocessing solution still works.

## What the testers and environment imply

- Both official notebooks expect code in their marked sections and a single completed notebook for submission. A tester success covers only its assertions. A has no automatic checks for the model definition or final two analyses. B’s architecture check counts parameters, so inspect ReLU, dropout order and evaluation mode yourself. Neither tester grades every requested plot, sample print or vectorization choice.
- Both papers warn that long loops (50+ iterations) over data may cost points. Comprehensions, map and np.vectorize do not qualify as vectorized solutions. Parsing and training/evaluation loops have stated exceptions. A also permits loops for the base merge. Use masks, broadcasting and axis reductions when the prompt asks for vectorization.
- The papers describe roughly 75–85% binary and 60–70% multiclass accuracy on their own data. Treat these as diagnostic ranges. Suspiciously high accuracy warrants a check for score leakage. For low accuracy, inspect label rules, feature order, output/loss pairing and input scale. Select checkpoints by **validation loss**. Weight each batch mean by its sample count so the short final batch contributes correctly.
- The pack describes Windows lab desktops, JupyterLab and no public internet. It lists the supplied course materials, hosted documentation and your own homework via a separate URL. Open docs.html links in a browser. Personal practice notebooks and dnd_utils.py are study aids at home. For an offline rehearsal, keep the pack, documentation and your homework at hand.
- Within the supplied course materials, Week 3 is a quick NumPy vectorization refresher, Week 5’s worksheet includes a histogram exercise, and Weeks 6–8 cover data loading and preparation. Practise locating one relevant example before a timed attempt.

**Local sources** · [Autumn pack instructions](../README.html) · [Documentation index](../docs.html) · [Official A](../dndmsc26spring_zh1a.ipynb) · [A tester](../dndmsc26spring_zh1a_tester.py) · [Official B](../dndmsc26spring_zh1b.ipynb) · [B tester](../dndmsc26spring_zh1b_tester.py) · [Existing exercise pack](../../../SPECIFIC%20EXERCISES/README.md)
