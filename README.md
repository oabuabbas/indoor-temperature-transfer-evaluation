# Indoor temperature transfer evaluation

Research code and aggregate results for a house-holdout indoor-temperature forecasting study. This repository is a research release candidate, not a claim of independent replication or a new forecasting algorithm.

## Scientific status
The original IDEAL primary comparison is unchanged. Revision v1 adds post-result descriptive heterogeneity analysis and a matched-support lag/nonlinear-model sensitivity. Prior results informed these additions. Associations are not causal and model configurations are limited.

## Data and execution
Original IDEAL data: https://doi.org/10.7488/ds/2836. Bremen: https://github.com/ComNets-Bremen/Office-Temperatures. Aalborg v3: https://doi.org/10.5281/zenodo.10673763.

No raw household time series, household metadata, home-level outputs, or downloaded third-party archives are distributed here. The scripts require the original data and regenerated intermediate files. This is not a one-command self-contained dataset. See REPRODUCIBILITY.md for the original preparation and evaluation sequence and limitations. The source room archive is about 10 GB. Review provider terms before downloading or redistributing derivatives.

After reproducing the primary pipeline, run `python revision_analysis_v1.py`, then `python verify_revision_v1.py`. Completed revision results are protected from overwrite. Use a fresh isolated directory for a rerun. The verifier checks hashes and stored arithmetic, not an independent model refit.

## Software
Primary recorded environment: Python 3.12.10, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1. Revision plotting requires matplotlib; threadpoolctl limits numerical threads. These versions document the environment and are not a portability guarantee. No deep-learning framework is required for revision v1.

## Rights and attribution
Upstream datasets retain their original licences and citations. No new blanket licence is assigned to third-party data. A licence for original code and finalized creator metadata remain to be confirmed by the author before a versioned archival release. No DOI has been assigned to this code package.
