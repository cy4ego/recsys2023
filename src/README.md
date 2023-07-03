# RecSys2023 Challenge
## How to use
### Prepare a parquet data from all csv files
Set the directory path according to your location where you saved challenge data, then run `python ./csv2pqt.py`

### Train and infer using challenge data
Run `python ./lgbm_train_and_infer.py`

## Result
This code alone get 6.187597 which is 13th for the leaderboard. In order to get 6.175644, you have to run this code four times. You have to set the configuration like following.   
- `N_COMPONENTS = 15`, `num_ctg_ns = list(range(70, 80)) + list(range(60, 64))`
- `N_COMPONENTS = 18`, `num_ctg_ns = list(range(70, 80)) + list(range(60, 64)) + [57]`
- `N_COMPONENTS = 17`, `num_ctg_ns = list(range(70, 80)) + list(range(60, 64)) + [57, 56]`
- `N_COMPONENTS = 17`, `num_ctg_ns = list(range(70, 80)) + list(range(60, 64)) + [57, 55]`

In addition to those four results, you have to run TabSurvery's testall.sh to get last result to ensemble them. See the `README_RECSYS2023.md` in TabSurvery
