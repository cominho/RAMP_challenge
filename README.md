# RAMP starting kit on prediction of stock returns ranking

[![Build Status](https://travis-ci.org/ramp-kits/variable_stars.svg?branch=master)](https://travis-ci.org/ramp-kits/variable_stars)

This challenge is based on a dataset available on Kaggle, described in detail in the following link : https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data. The dataset provides 5minutes-by-5minutes information on NIFTY 50 assets.
The task is to build a model that can rank these assets based on their returns using advanced machine learning techniques. By predicting returns accurately, the model aims to identify the most promising assets for investment.
This challenge is interesting because accurate asset ranking can help investors make informed decisions and better manage risks in volatile financial markets.
The repository includes a data science pipeline as a baseline, with a custom NDCG scoring function for evaluating rankings. 


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](stock_ranking_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
