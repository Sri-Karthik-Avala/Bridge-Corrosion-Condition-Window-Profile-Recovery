# Bridge Corrosion Condition Window Profile Recovery

| | |
| --- | --- |
| Final rank | #6 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 8 |
| Last submission | 2026-08-02 |

## Problem statement

### Overview

Bridge inspection photos often contain small, irregular corrosion regions spread across beams, fasteners, joints, and concrete-steel boundaries. In this task, each row gives a transformed inspection window and asks you to reconstruct a compact corrosion-condition profile over an 8 by 8 grid.

For every image, predict which grid cells contain annotated corrosion, which cells correspond to fair, poor, or severe condition regions, coarse area bins for each condition state, and an eight-row vertical condition profile. The target is not a single label for the image; it is a structured record describing where corrosion appears and how the visible condition states are distributed.

### Dataset

Files:

- `train.csv`: 1,108 labeled examples.
- `test.csv`: 467 held-out examples.
- `sample_submission.csv`: A weak schema example with the required columns.
- `images/`: 1,575 JPEG inspection windows referenced by `image_path`.

Columns in `train.csv`:

- `id` (string): Row identifier.
- `image_path` (string): Relative path to the transformed inspection-window JPEG.
- `packet_json` (JSON string): Input constraints. It contains `image_size` as `[288, 288]`, `grid_shape` as `[8, 8]`, and `condition_states` as `["fair", "poor", "severe"]`.
- `answer_json` (JSON string): Ground-truth corrosion-condition profile.

Columns in `test.csv`:

- `id` (string): Row identifier.
- `image_path` (string): Relative path to the transformed inspection-window JPEG.
- `packet_json` (JSON string): Same structure as in `train.csv`.

The `answer_json` object has exactly these fields:

- `affected_cells` (array of strings): Grid cells containing any annotated corrosion. Cell tokens use `rRR_cCC`, for example `r03_c05`.
- `fair_cells` (array of strings): Affected cells containing fair-condition corrosion.
- `poor_cells` (array of strings): Affected cells containing poor-condition corrosion.
- `severe_cells` (array of strings): Affected cells containing severe-condition corrosion.
- `area_bins` (object): Integer bins from 0 to 9 for `fair`, `poor`, and `severe`, representing coarse annotated area for each state.
- `row_condition_profile` (array of 8 integers): One value per grid row. `0` means no corrosion in that row, `1` fair only, `2` poor only, `3` severe only, and `4` mixed states.

All cell arrays must contain valid cells from the released 8 by 8 grid. The state-specific cell arrays must be subsets of `affected_cells`.

### Evaluation

The score is the mean row score over the held-out test rows.

For a set-valued field, F1 is:

`F1 = 2 * precision * recall / (precision + recall)`

with `precision = |predicted intersection true| / |predicted|` and `recall = |predicted intersection true| / |true|`. If both sets are empty, F1 is 1. If exactly one set is empty, F1 is 0.

For each row:

- `affected_score = F1(predicted affected_cells, true affected_cells)`.
- `state_score` is the mean F1 for `fair_cells`, `poor_cells`, and `severe_cells`.
- `bin_score` is the mean over `fair`, `poor`, and `severe` of `max(0, 1 - abs(predicted_bin - true_bin) / 4)`.
- `profile_score` is the fraction of the 8 `row_condition_profile` positions that match exactly.

The raw row score is:

`0.34 * affected_score + 0.30 * state_score + 0.20 * bin_score + 0.16 * profile_score`

Malformed JSON or invalid values score 0 for that row. The submitted CSV must still have exactly the required columns, exactly one row per test id, and no missing, extra, or duplicate ids.

After the component-weighted raw row score is computed, the grader applies a strictness transform to reduce credit for broad base-rate guesses: `row_score = raw_row_score ^ 4.0`. Perfect rows still score `1.0`, malformed rows score `0.0`, and partial rows must be close across several fields to retain substantial credit.

### Submission

Submit a CSV with exactly two columns:

- `id` (string)
- `answer_json` (JSON string)

Example:

```
id,answer_json
corr_example_01,"{""affected_cells"":[""r02_c03"",""r02_c04""],""fair_cells"":[""r02_c03""],""poor_cells"":[""r02_c04""],""severe_cells"":[],""area_bins"":{""fair"":2,""poor"":1,""severe"":0},""row_condition_profile"":[0,0,4,0,0,0,0,0]}"
corr_example_02,"{""affected_cells"":[""r05_c01""],""fair_cells"":[],""poor_cells"":[],""severe_cells"":[""r05_c01""],""area_bins"":{""fair"":0,""poor"":0,""severe"":1},""row_condition_profile"":[0,0,0,0,0,3,0,0]}"
```

### What Not To Use

- Do not use any files outside the supplied dataset package.
- Do not use external image collections, annotation collections, search engines, or lookup services.
- Do not hard-code row ids, image names, or answers.
- Do not manually label the held-out images.
- Do not use hosted vision APIs or pretrained external checkpoints.
