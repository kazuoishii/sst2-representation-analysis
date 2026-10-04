# SST-2 BERT representation analysis

Code and five-seed results for layerwise analysis of frozen `bert-base-uncased` representations on SST-2. This package covers representation analysis only. Recursive knowledge distillation is a separate experiment.

## Experimental design

- Frozen pretrained BERT in evaluation mode; no encoder fine-tuning.
- CLS representations from the embedding state (layer 0) and 12 transformer layers.
- Each seed (42–46) samples 5,000 training examples without replacement from 67,349 SST-2 training examples.
- Evaluation uses the same 872 development examples. This is the development/validation split, not the hidden-label test split.
- StandardScaler followed by LogisticRegression (liblinear, max_iter=5000), independently fitted at each layer.
- Batch size 32; maximum sequence length 128; execution selects Apple MPS when available, otherwise CPU. The original script does not select CUDA.

The five runs vary the training subset and probe seed, not independently trained BERT encoders. CKA, effective rank, and geometric distance use the shared development representations and are identical across the supplied seeds. SD therefore describes probe/subset variability, not encoder-training variability.

## Environment

Python 3.11.16 and package versions in `requirements.txt` were reported from the user's Mac environment on 2026-10-04. They document the environment inspected on that date; historical training versions and a clean installation of these pins have not been independently verified.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The model identifier is not pinned to a Hugging Face commit in the original scripts. Bitwise reproduction of encoder outputs is therefore not guaranteed across model revisions, hardware, or library versions.

## Retrieve data

From the repository root:

```bash
curl -fL https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py -o /tmp/download_glue_data.py
.venv/bin/python /tmp/download_glue_data.py --data_dir data --tasks SST
shasum -a 256 -c SHA256SUMS-data.txt
```

The retrieval and extraction commands were successfully executed on the user's Mac. Both TSV hashes matched the files used for the supplied experiment. The historical download URL is unknown. The public script URL uses the moving `master` branch; hashes provide the data identity check. Dataset bytes are not included in this package.

## Recompute summaries and figures (no BERT inference)

The five per-seed CSVs are included. To regenerate a summary and compare it with the supplied reference:

```bash
.venv/bin/python aggregate_sst2_results.py --reference sst2_multiseed_summary.csv
```

This writes `sst2_multiseed_summary_rebuilt.csv` and refuses to overwrite an existing output. For the original plot script, the supplied reference summary is already available:

```bash
.venv/bin/python plot_sst2_multiseed.py
```

The plot writes PDF and PNG files in the working directory. Layer 0 is omitted from the main figure.

## Rerun representation analysis

```bash
for seed in 42 43 44 45 46; do
  SEED=$seed .venv/bin/python run_sst2_multiseed.py || break
done
```

The experiment overwrites the corresponding per-seed CSVs. Run in a separate copy of this repository to preserve the supplied results. `run_sst2_representation.py` is the original single-seed (42) script; the multi-seed script is the recommended entry point.

## Metrics and indexing

| Column | Definition |
| --- | --- |
| Q_accuracy | Development-set linear-probe accuracy, fraction |
| effective_rank | exp(entropy) of normalized singular values of centered development representations; singular values, not squared singular values |
| CKA | Centered linear CKA between layer l and l+1 |
| Delta_CKA | 1 minus adjacent-layer CKA |
| Delta | Frobenius norm of H(l+1)-H(l), divided by the Frobenius norm of H(l) |
| delta | max(Q(l)-Q(l+1), 0) |

Transition metrics at row l refer to l → l+1 and are missing at layer 12. Layer 0 CLS is constant across examples; its effective rank and CKA are degenerate and should not be interpreted as meaningful contextual geometry. Probe accuracy is an operational task measure, not a direct measurement of Shannon information or proof of irreversible information loss.

The summary reports mean accuracy, sample SD (ddof=1), and mean/SD of the per-seed positive accuracy decreases. Mean positive decrease is not generally the positive decrease of mean accuracy. Shared geometry columns are retained once after checking equality across seeds.

## Supplied results

Peak mean probe accuracy occurs at layer 11: 0.831422 ± 0.016096 (sample SD, five seeds). Layer 12 accuracy is 0.816284 ± 0.006090. These results describe the specified frozen encoder and probe setup.

## Validation and publication status

Python syntax, CSV schemas, five-seed aggregation, and numeric agreement with the supplied summary were checked. The full BERT inference experiment was not rerun in the packaging environment. Code and reproducibility materials are publicly available at https://github.com/kazuoishii/sst2-representation-analysis.

## License and authorship

Software and accompanying repository documentation: MIT License, Copyright (c) 2026 Kazuo Ishii. This software license does not license the associated manuscript, the SST-2 dataset, or pretrained BERT weights. Original research-result CSVs and figures are supplied for reproducibility; no separate content license has been assigned to them.

Software citation author and copyright holder: Kazuo Ishii.

Associated manuscript authors, in the order shown in the supplied final TeX:

1. Kazuo Ishii — Department of Applied Information Engineering, Faculty of Engineering, Suwa University of Science, Japan.
2. Bishnu Prasad Gautam — same affiliation.
3. Javaid Saher — Department of Information Engineering, Kanazawa Gakuin University, Japan.

Correspondence: kishii@rs.sus.ac.jp.

## Citation

Ishii, K., Gautam, B. P., and Saher, J. (2026). *Geometric Deformation Does Not Imply Task-Relevant Information Degradation in Neural Representations*. Unpublished manuscript.

Zenodo and arXiv publication is planned. No Zenodo DOI or arXiv identifier has been assigned yet. Update this section and `CITATION.cff` when identifiers are available. The citation author names follow the supplied TeX exactly. `CITATION.cff` distinguishes the sole software author from the three manuscript authors.

Please also cite GLUE, the Stanford Sentiment Treebank, and BERT as appropriate. Academic citation is requested; MIT licensing itself requires retention of the copyright and license notice, not a scholarly citation.

