# AI Image Detection: Which Meta-learners Help Expert Models Generalise to Unseen Generators?

SC4001 group project, group 40. Can a combination of generator-specific detectors ("experts") spot AI-generated
images from a generator none of them has seen? On a 176,000-image subsample of GenImage we train seven experts per
backbone (linear probes on frozen CLIP ViT-L/14 features and fine-tuned ResNet-50s, plus a ResNet-18 backup pool),
combine their scores with eight combiners, and compare them under leave-one-generator-out evaluation with the best
single expert and a single generalist detector. The report source is in `report/main.tex`.

The `input/` folder (images, features, trained models, results; ~10 GB) is **not** included because of the upload
limit. The notebooks are submitted **with their outputs**, so the main tables and figures of the report (Tables 2, 4
and 6-8, the heatmaps behind Tables 3 and 9, and the robustness checks of Section 2.3) can be read from them without
rerunning anything. The steps below recreate `input/` from scratch.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | Download, sampling, preprocessing and the manifest (`data/config.py` holds all dataset settings) |
| `clip/` | CLIP feature extraction, CLIP experts (logistic regression) and their scores |
| `resnet/` | ResNet experts and generalists: training, temperature calibration, scoring |
| `meta/` | Leave-one-out evaluation (`loo.py`), meta-learners (`learners/`, one file each), CLIP generalist, paired bootstrap |
| `common/` | Device selection, logging, evaluation helpers |
| `notebooks/` | Entry points. Each notebook is configured by the settings in its first code cell |
| `report/` | LaTeX source of the report |

## 1 · Dependencies

Python 3.10 (other 3.10+ versions should work). A GPU is optional: `common/device.py` uses Apple `mps`, then `cuda`,
then `cpu`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CLIP weights (`openai/clip-vit-large-patch14`) and the ImageNet ResNet weights are downloaded automatically
on first use.

## 2 · Data preparation

Source: the GenImage Arrow mirror [`nebula/GenImage-arrow`](https://huggingface.co/datasets/nebula/GenImage-arrow)
on the Hugging Face Hub. If the Hub asks for authentication, run `hf auth login` or set `HF_TOKEN`.

Run **`notebooks/download_data.ipynb`**. It downloads shards one at a time to `SCRATCH_DIR`
(default `/tmp/genimage_scratch`, deleted after use), draws a seeded sample (`SEED = 4001`), converts to RGB,
upscales images with a short side below 256 px (bicubic), takes a 256 × 256 center crop, re-encodes reals and fakes
alike as JPEG q95 and writes

```
input/genimage_<hash>/
  manifest.csv                     one row per image: file, pool, generator, label, original size
  expert_train/ expert_val/ meta_train/ meta_val/ test/   <generator>/<label>/...jpg
```

Per generator and label: 5000 `expert_train`, 1000 `expert_val`, 2000 `meta_train`, 1000 `meta_val`, 2000 `test`
(8 generators → 176,000 images). `test` is drawn from GenImage's test split, all other pools from its train
split (report Section 2.2, Appendix A for the shard sampling). The folder name is a hash of the settings in `data/config.py`, so with
unchanged settings it is `input/genimage_d8889a9c`, the one used in the report. The download resumes where it
stopped and restarts itself after network errors. Set `PILOT = True` in `data/config.py` for a ~1/5-scale run.

## 3 · Reproducing the main results

Run the notebooks in this order. Every step saves its output inside the dataset folder and skips work that is
already done, so it is safe to stop and rerun. Most training notebooks have `LIMIT = 100` for a quick test run
that writes to separate folders; use `LIMIT = None` for the real run.

| # | Notebook | What it produces |
|---|---|---|
| 1 | `notebooks/download_data.ipynb` | The image sample and `manifest.csv` |
| 2 | `notebooks/clip/extract_clip_features.ipynb` | CLIP ViT-L/14 features of every image (`features/clip_vit-l14`) |
| 3 | `notebooks/clip/clip_experts.ipynb` | 7 CLIP experts and their scores (`features/clip_vit-l14_experts`) |
| 4 | `notebooks/resnet/resnet_experts.ipynb` with `ARCH = "resnet50"` (main pool), then `ARCH = "resnet18"` (backup pool) | 7 ResNet experts per architecture and their scores (`features/resnet50_experts`, `features/resnet18_experts`) |
| 5 | `notebooks/resnet/resnet_generalists.ipynb` with `ARCH = "resnet50"`, then `"resnet18"` | 8 generalists per architecture (one per fold) and their scores |
| 6 | `notebooks/meta_learners.ipynb` | All meta-learners on all pools, leave-one-out, plus the paired bootstrap |

In step 6, `run_all(eval_pool=EVAL_POOL)` writes `results/meta_learners_test.csv` and the held-out scores the
bootstrap reads. The CLIP generalist is fitted inside `run_all`; the ResNet generalists are picked up from step 5.

Main settings (first code cell of `meta_learners.ipynb`):

- `EVAL_POOL = "test"` gives the final numbers in the report. Use `"meta_val"` for development; `test` was only
  evaluated once, after all choices were frozen.
- `MERGE_SIBLINGS = True` holds out SD (1.4 + 1.5) and Wukong together (6 folds), as in the main tables.

### Where each result in the report comes from

| Report | Notebook |
|---|---|
| Section 2.3, upscaling confound (8.7% upscaled reals, probe AUROC 0.965, control 0.907, BigGAN expert on upscaled reals) | `notebooks/report/confound_check.ipynb` |
| Section 2.3, JPEG check | `notebooks/report/jpeg_check.ipynb` (needs the ResNet-50 experts) |
| Section 3.3 / Table 2, expert AUROC per generator | `notebooks/evaluation.ipynb` (`ARCH = "resnet18"` for the backup pool numbers) |
| Section 3.4 / Appendix D (Table 7), ResNet-50 generalists | `notebooks/resnet/resnet_generalists.ipynb` |
| Section 5 / Table 3, Appendix F (Table 9), main results | heatmaps in `notebooks/meta_learners.ipynb` (columns: `clip`, `resnet18`, `resnet50`) |
| Tables 4 and 8, paired bootstrap | last cell of `notebooks/meta_learners.ipynb` |
| Appendix B, linear probe vs. kNN | `notebooks/clip/probe_vs_knn.ipynb` |
| Appendix C (Table 6), ResNet-50 expert training | `notebooks/resnet/resnet_experts.ipynb` |

Some numbers are computed from the saved results but not shown in a notebook output: the per-fold table (Table 5;
per-fold rows are in `results/meta_learners_test.csv`), the best-expert-in-hindsight row of Table 3
(`expert_best_auc` column of the same CSV), the seed standard deviations and the standardised-logits
variant (Section 4.3, `scale=True` rows of the same CSV), the seven-fold comparison (`MERGE_SIBLINGS = False`), and the
meta_val bootstrap (`EVAL_POOL = "meta_val"`).

`notebooks/clip/clip_test.ipynb` and `notebooks/clip/probe_one_generator.ipynb` are early sanity checks and are not
needed for the results.