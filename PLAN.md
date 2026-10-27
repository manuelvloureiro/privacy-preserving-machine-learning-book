# Plan: simpler, consistent books with one reading order

Status on 2026-10-27. Work items, largest saving first within each step. Word counts are estimates from reviews of every book and note; each item says what to keep, so no proof is lost. Delete this file when the last item is done.

## Done

- The exercise book is gone: `book-exercises/`, its 48 solution scripts and tests, `tools/xrefs.py`, `filters/xrefs.lua`, the Makefile target, the structure, prose and notation rules for exercise pages, and about 270 exercise pointers in the full edition (the last-line **Practice** link of every chapter and every inline `[Exercise N](…)`). AGENTS.md now says the books have no exercises.
- The landing page (`site/index.html`) gives the reading order below, and the full edition's preface points to the course first.
- Steps 4–7 for the practitioner edition, the course, the OpenDP book and five notes (2026-10-27), together with a paragraph rewrite of every page to PROSE.md rules 19–25:
  - Step 4.1, 4.2 and 4.3 are done. `10-calibrate-dp-sgd` owns the Adult calibration, cross-check and statement. `13-train-dp-sgd` and `12-release-governance` link to it. `10-risk-assessment` is a decision page. `12-release-governance` is the one home of the statement and claim review (`#sec-relgov-statement`, `#sec-relgov-review`). The δ rule's one home is `01-definition#sec-dpdef-small-delta`. The stale link text is fixed.
  - Step 5 is done. Each course property is proved once: the [0,1]-functions fact is `lem-rr-functions`, at its first use, and `lem-adp-functions` adds δ. There is one definition of adaptive composition (`def-lap-adaptive`), and the shuffle and mean-estimation lemmas cite their originals. History and asides are cut, references are one sentence each, and part pages are one paragraph plus the table.
  - Step 6 is done. Chapter 03 owns the first count. The margin lives in chapter 02. "36 quarters" is a clause plus a link. The wrong pointers and the Context API recap are fixed.
  - Step 7.2 (the trade-off note's tables) and 7.3 are done.

## The reading order

The books are read in this order, and every preface and the landing page say so:

1. **The course** (`book-course/`): the foundations. Self-contained, every proof, little text.
2. **The full edition** (`book-full/`): the depth. Mathematics, attacks, foundations, accounting, private ML, with appendices for each cited step.
3. **The practitioner edition** (`book-practitioner/`): the practice. The same parts without proofs, with workflows on real libraries. A reader who wants results rather than proofs starts here.
4. **The OpenDP book** (`book-opendp/`): one library end to end.
5. **Study notes**, each after the book it extends. Concentrated privacy comes after the course's approximate-DP and advanced-composition chapters. The Python libraries note and the trade-off note come after the practitioner edition. AIM comes after the OpenDP book's Part 5. Datasets is for designing an experiment.

What this order asks of each book:

- **Course preface:** say in one sentence, without a link (the course links to no book), that the full edition goes deeper and the practitioner edition shows the practice.
- **Full edition:** a reader who has done the course can skim Part 3. The Part 3 page should say which chapters go beyond the course: privacy profile, discrete mechanisms, trust models and mean estimation.
- **Practitioner and OpenDP prefaces:** name the step before them, the full edition for the why. The OpenDP book should link to the practitioner workflow it extends, `book-practitioner/part03-dp-foundations/12-release-statistics.md`, from chapter 02 or 03.

## Step 1. Claims left without support by the exercise removal (full edition)

About 90 sentences stated a result and pointed to an exercise for its proof or computation. They now stand alone. For each one, either prove it on the page in at most a few lines, compute it in the chapter's script and quote it in Code, cite a paper, or cut the sentence. The list is in the appendix of this file.

## Step 2. The full edition's appendices (about 37k of 143k words)

1. **Retire core appendices 01–04 (about 31k words).**
   - Since the conversion, these re-derive step for step what the chapters prove:
     - Laplace, the classic and exact Gaussian, the exponential mechanism, randomized response and sensitivity: `thm-lg-*`, `thm-sel-em`, `thm-dpdef-*`.
     - Basic, advanced, Gaussian and RDP composition: `thm-dpprop-basic`, `thm-comp-advanced`, `thm-gdp-composition`, `thm-rdp-composition`.
     - All of DP-SGD accounting: `lem-dpsgd-sensitivity`, `thm-sub-*`.
     - The testing, posterior, RDP, zCDP and GDP conversions.
   - Move the few unique parts into the chapter that uses them:
     - the tail rule σ_tail and its two regimes (01 §tradeoffs) into `03-laplace-gaussian` or `09-choosing-epsilon`;
     - the accountant comparison (02 §compare) into `09-choosing-an-accountant`;
     - the zCDP inverse formula (04 §zcdp step 8) into `03-concentrated-dp`;
     - the replace-one 2C clipping case into `02-dp-sgd`.
   - Renumber the appendices (A–J), fix `_quarto.yml`, and update every `@sec-apx-cal-*`, `@sec-apx-ccomp-*`, `@sec-apx-sgd-*` and `@sec-apx-conv-*` link.
   - Drop `code/appendix-derivations/apx_{calibration,composition,dp_sgd,conversions}.py` and their tests once nothing quotes them.
   - Rewrite AGENTS.md "Appendices of derivations": there are no core appendices; an appendix proves only what a chapter cites or sketches, and never re-derives a result a chapter proves.
2. **Remove partial overlaps in 05–14 (about 5.8k words).**
   - 11 comp-fdp/clt/sgd-clt: cut the opening steps that repeat `thm-gdp-*`.
   - 12 amp-clones steps 1–6 repeat `thm-shuf-clone-reduction`.
   - 13 learn-user steps 2–5 repeat `prp-ulvl-pair-moment`.
   - 14 audit-onerun steps 1–3 repeat `thm-audit-onerun`.
   - 13 learn-fairness's last step repeats `thm-fair-floor`.
   - 12 amp-poisson's "bonus" repeats `thm-sub-poisson`.
   - 06 tail-renyi-kl steps 5–6 repeat `thm-div-renyi`.
   - Optionally, 05 prob-convolution re-proves results by another route.
3. **Trim the appendix layout (about 12–15k words).**
   - "What the chapter uses" averages 300 words of narration: make it the statement plus one sentence.
   - "The idea" often restates the derivation: keep it only where it adds a small concrete case.
   - "Checking it" repeats numbers already in the text: keep the check and one figure.
   - Update the `skeletons.appendix` headings in `tools/structure.toml` to match.
4. **Fix text about other pages that has gone stale.**
   - 10-mechanisms §svt quotes a selection-chapter sentence that no longer exists.
   - 10 §gumbel says the chapter writes β, but it writes κ.
   - 12-amplification and 11-composition list "gaps" in other chapters.
   - State the result instead of narrating what a chapter does or omits.

## Step 3. One proof per lemma in the full edition (about 3–4k words)

Keep the canonical copy listed first, and replace each other proof with a one-sentence restatement and a link.

| Result | Canonical | Re-proved in |
| --- | --- | --- |
| Best/bad event, hockey-stick ≤ Pr[L > ε] | `lem-div-best-event`, `prp-div-hockey` | `thm-dpdef-divergence`, `lem-dpprop-tests`, `lem-lg-bad-event`, `lem-comp-delta`, `thm-num-delta` step 2, `prp-conv-mironov` step 1, `thm-ht-tv`, `thm-ht-bayes`, `prp-threat-bayes`, `thm-epsc-curve` step 4 |
| Gaussian privacy loss N(μ²/2, μ²) | `thm-dist-shift` | `prp-div-gauss`, `thm-ht-gaussian`, `lem-lg-gaussian-loss`, `lem-gdp-projection`, `prp-ulvl-pair-moment`, `thm-sub-rdp-gaussian` step 5 |
| Exact Gaussian δ(ε) | `prp-div-hockey-pairs` | `thm-lg-analytic` step 3, `cor-gdp-profile` |
| Laplace loss by the triangle inequality | `thm-dist-shift` | `prp-div-laplace`, `thm-lg-laplace` |
| RDP → DP conversion | `thm-rdp-conversion` | `prp-conv-mironov`, `prp-cdp-tail` step 2; the order optimisation in `cor-rdp-gauss-opt` and `thm-cdp-conversion` |
| Loss of a composition is a sum | `lem-dpprop-add` | `lem-comp-sum`, `thm-dpprop-basic` steps 1–3 |
| Post-processing | `thm-div-dpi` | `thm-dpprop-post` step 4, `lem-threat-processing` and `lem-attr-processing` (the same simulation twice in Part 2) |
| DP-SGD accountant | `thm-sub-dpsgd` | `thm-dpsgd-privacy` (the same theorem) |
| Change of measure | `lem-div-change-of-measure` | `lem-ht-change-of-measure` |

Also unify the symbols that change between chapter and appendix: α/β in the chapters, FPR/FNR in the appendices; β against κ for the Gumbel scale. Put the choice in NOTATION.md and `notation.toml`.

Delete the part `README.md` files. Readers never see them, and they repeat the part pages. If GitHub browsing needs a README, keep one line that links to the part page.

## Step 4. Practitioner edition (about 4.5k words, or 10.5k with Part 1)

1. **Give each step of the DP-SGD trio one owner (about 1.5k).**
   - The same Adult run (σ = 0.973, ε ≈ 3.98) is calibrated, cross-checked and written up three times.
   - `10-calibrate-dp-sgd` owns calibration, the cross-check and the statement.
   - `13-train-dp-sgd` links to it and keeps training, tuning and utility.
   - `12-release-governance` keeps evidence, regulation and review.
2. **Make `10-risk-assessment` a decision page (about 1.45k).** It should hold thresholds, verdicts and the write-up, and link to the runs in 02, 05, 07 and 08 instead of repeating their tables. Keep the canary/LoRA run once, in `14-private-llm-finetuning`.
3. **Give each repeated rule one home (about 1.2k).**
   - The privacy statement and claim review go in `12-release-governance`; `09-choosing-an-accountant` keeps only choosing and matching an accountant.
   - The δ ≤ 1/n^1.1 rule gets one home, `01-definition#sec-dpdef-small-delta`, and a clause plus a link everywhere else.
   - Fix the stale link text "Report and review an ε claim" on six pages.
4. **One running example for DP-SGD.** `02-dp-sgd` still uses the synthetic 10,000 records while the workflows use Adult. *Not done:* only 02's budget table has an Adult counterpart. Its clipping, norm-drift, microbatching and fixed-batch numbers, and pages 06–09, would need new Adult runs in `code/`.
5. **Part 1.**
   - Three of its eight pages have no incoming link, and 06-optimization is rate theory.
   - Keep `04-hypothesis-testing`, perhaps as a Part 2 page. Fold the few divergence facts into `01-definition` and `01-renyi-dp`, and link to the full edition's Part 1 for the rest.
   - This needs AGENTS.md's "every topic appears" rule relaxed first.
6. **Later:** rewrite the topic chapters in job order (decision → default → call → pitfall) instead of the full edition's teaching order.

## Step 5. Course (about 10k words)

1. **Prove each property once (about 950 words).**
   - Post-processing and composition are proved for pure DP in `02-laplace-mechanism`, then again with δ in `03-approximate-dp`: make the second "the same proof with +δ".
   - The [0,1]-functions lemma is proved three times; `lem-adp-functions` becomes the only proof.
   - `def-ac-adaptive` is the third definition of adaptive composition.
   - `lem-mean-gauss-tail` repeats `lem-adp-gaussian-facts`.
   - `lem-shuf-renyi-product` and `prp-shuf-gaussian` repeat `lem-dpdl-compose` and `lem-dpdl-gaussian`.
2. **Make the text light, as the course promises (about 6–8k).**
   - Cut history and asides in `01-attempts`, `01-privacy-violations`, `03-private-deep-learning`, `01-local-dp` and `02-shuffle-and-central-deployments`, which are 70–80% prose.
   - Bring References down to one sentence per source.
3. **Part pages (about 1k).** Keep one framing paragraph plus the table; the per-chapter paragraphs repeat the chapter openings.

## Step 6. OpenDP book (about 1.5k words)

- The same total-count release ("within 539.7") appears in chapters 02, 03 and 07. Chapter 02 should stop at the context, and chapter 03's count becomes the first LFS release.
- The margin is introduced in 02 and again in 03:17, which wrongly says the first chapter's context had none.
- The privacy unit "36 quarters" is restated about 12 times: cut each to a clause plus a link to `#sec-ctx-context`. Fix the pointer at `04-grouping.md:15`, which credits the wrong chapter with introducing the dataset.
- Drop the Context API recap in chapter 03, and trim the AIM narrative in chapter 17 (about 150 words).

## Step 7. Study notes (about 2.5k words)

1. **Merge the two trade-off notes** into one note with shared Steps 1–3, a model route and a synthetic-data route, and one decision.
   - About 1.2k words of the synthetic note repeat the first.
   - It refers to "the companion note" 10 times, which breaks the rule that notes stand alone.
   - Its results are still `RESULTS-*` placeholders.
   - Do this once the synthetic sweep has run.
2. **Give the trade-off notes the section-map and symbol tables** the note rule requires, and make `skeletons.note` in `tools/structure.toml` enforce them. *The tables are done for `privacy-utility-tradeoff`.* The synthetic note and the enforcement wait for 7.1.
3. **Trim the rest.**
   - dp-datasets §synthetic re-argues the AIM note's "why marginal generators win" (about 500 words). Its §choose depends on "the companion note".
   - python-privacy-libraries previews itself four times, and its "four duties" overlap its "five checks" (about 450 words).
   - aim's "This section says…" paragraphs repeat its map table, and "What the note proves and cites" repeats Sources (about 300 words).

## Step 8. Guards so it stays consistent

- **`make duplicates`:** flag paragraph pairs with high word-shingle overlap within a book and across books (the script used for these reviews), and run it in `make check` with an allow-list.
- **A constants check:** test that the course and the full edition quote the same values for shared constants, such as the classic Gaussian σ, the advanced-composition bound and the moments-accountant bound. The course cannot link, so drift is the risk.
- **A canonical-proof list:** a short table in AGENTS.md of the lemmas above with their one home, so a new chapter cites instead of re-proving.

## Appendix: claims to settle from Step 1

Line numbers as of 2026-10-27, all under `book-full/`.

**Part 1.**
- `01-distributions`: 187, 204, 481.
- `02-concentration`: 176, 234.
- `03-divergences`: 392, 561.
- `04-hypothesis-testing`: 140, 190, 204 (209 readings), 320, 373 (2,995 negatives).
- `05-information-theory`: 397.
- `06-optimization`: 119, 284, 326, 377, 452.
- `07-stability-generalization`: 176, 198, 257.
- `08-crypto-basics`: 159, 340.

**Part 2.**
- `01-threat-models`: 281, 298.
- `02-membership-inference`: 95, 143, 260.
- `03-attribute-inference`: 63, 167.
- `04-property-inference`: 102, 138, 231, 282.
- `05-memorization-extraction`: 79.
- `06-gradient-inversion`: 147, 180, 182, 201, 277, 318.
- `07-linkage-reidentification`: 31, 92, 145.
- `08-syntactic-anonymity`: 91, 177, 258.
- `09-reconstruction`: 50, 84, 199, 284, 313, 357.

**Part 3.**
- `01-definition`: 41, 51.
- `02-properties`: 118.
- `03-laplace-gaussian`: 55, 115, 230.
- `04-composition`: 237, 250.
- `05-selection`: 39.
- `06-query-release`: 159.
- `07-sensitivity`: 163.
- `08-discrete-mechanisms`: 117, 244, 353.
- `09-trust-models`: 186.
- `10-lower-bounds`: 198, 328.

**Part 4.**
- `01-renyi-dp`: 389.
- `02-rdp-conversions`: 111.
- `03-concentrated-dp`: 129, 186, 253.
- `04-gaussian-dp`: 33, 221.
- `05-subsampling`: 132.
- `08-adaptive-composition`: 275.
- `09-choosing-an-accountant`: 152, 272.

**Part 5.**
- `01-dp-erm`: 129.
- `02-dp-sgd`: 320.
- `03-hyperparameter-tuning`: 137.
- `04-matrix-mechanisms`: 173.
- `06-public-data`: 298.
- `07-user-level`: 53.
- `08-disparate-impact`: 236, 293.
- `09-choosing-epsilon`: 314.
- `10-auditing`: 169, 246.
- `11-adaptive-data-analysis`: 53, 88, 129, 134.
