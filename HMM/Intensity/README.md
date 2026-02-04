# Hidden Markov Model Analysis of Photobleaching

## Overview

The HMM analysis code expects a **whitespace-delimited text file** (not comma-separated CSV) containing time-series photon counting data with binary pre-selection flags. The binary label was a value from a machine-learning algorithm that identified whether photobleaching patterns were identified. That algorithm is not included here. If you do not have a pre-sorting algorithm, you can simply add binary values of `1` to each row of data.

* `PBIntensity321.HMM.py` original, fully-functional code.
* `PBIntensity321.HMM_Claude.py` refactored code by Claude AI. Fully-functional code. Details about refactoring in `REFACTORING_NOTES.md`.

## File Structure

### Raw File Format

The file should be structured as follows (whitespace-delimited):

```
<row_label>  <binary_1>  <binary_2>  <binary_3>  ...  <binary_N>
<time_0>     <value_1>   <value_2>   <value_3>   ...  <value_N>
<time_1>     <value_1>   <value_2>   <value_3>   ...  <value_N>
<time_2>     <value_1>   <value_2>   <value_3>   ...  <value_N>
...
<time_M>     <value_1>   <value_2>   <value_3>   ...  <value_N>
```

### Components

| Component | Description | Type | Example |
|-----------|-------------|------|---------|
| `<row_label>` | Header for the first column (ignored after loading) | String/Numeric | `time` or `index` |
| `<binary_i>` | Binary flag (0 or 1) indicating whether to process trajectory i | Integer (0 or 1) | `1` |
| `<time_j>` | Identifier for time point j (can be actual time or index) | Numeric | `0.0`, `0.1`, `0.2`, etc. |
| `<value_i>` | Intensity/count value for trajectory i at a given time point | Float | `15.3`, `22.7`, etc. |

### Dimensions

- **Rows**: `M + 1` (1 header row + M time points)
- **Columns**: `N + 1` (1 label column + N trajectories)

## Example File

```
Index    1      0      1      1      0
0.0      12.5   8.3    15.2   10.1   9.7
0.1      13.1   8.1    15.8   10.3   9.5
0.2      12.8   7.9    14.9   10.0   9.8
0.3      25.4   8.2    28.1   22.5   9.6
0.4      24.9   8.0    27.8   22.1   9.4
0.5      25.1   8.1    28.2   22.3   9.7
...
```

### Interpretation

In the example above:
- **5 trajectories** (columns 2-6)
- **Binary flags**: `[1, 0, 1, 1, 0]`
  - Trajectory 1: **PROCESSED** (binary = 1)
  - Trajectory 2: **SKIPPED** (binary = 0)
  - Trajectory 3: **PROCESSED** (binary = 1)
  - Trajectory 4: **PROCESSED** (binary = 1)
  - Trajectory 5: **SKIPPED** (binary = 0)
- **Time points**: 0.0, 0.1, 0.2, ... (typically 0.1 second bins)
- **Values**: Photon counts or intensity measurements at each time point

## Data Processing Flow

```
┌─────────────────────────────────────────┐
│  Raw Input File                         │
│  (whitespace-delimited)                 │
│                                         │
│  Row 1: Label + Binary Flags (0 or 1)  │
│  Row 2+: TimeID + Intensity Values      │
└──────────────┬──────────────────────────┘
               │
               │ pd.read_csv(..., delim_whitespace=True, 
               │             index_col=0).T
               ▼
┌─────────────────────────────────────────┐
│  After Loading & Transpose              │
│                                         │
│  Rows: Time points                      │
│  Columns: Trajectories                  │
│  Column Names: Binary flags             │
└──────────────┬──────────────────────────┘
               │
               │ Filter: Keep only where binary == 1
               ▼
┌─────────────────────────────────────────┐
│  Processed Trajectories                 │
│                                         │
│  Each column → HMM Analysis             │
│  - Change point detection               │
│  - State optimization                   │
│  - Path reconstruction                  │
└─────────────────────────────────────────┘
```

## Key Requirements

### ✅ Required
- **Whitespace-delimited** (spaces or tabs between values)
- **Numeric values** throughout (except optional first cell)
- **Binary flags** in first row (0 or 1 only)
- **Consistent number of columns** across all rows
- **At least one trajectory** with binary flag = 1

### ❌ Not Supported
- Comma-separated values (CSV format)
- Header row with column names
- Missing values (NaN)
- Non-numeric data in value cells
- Inconsistent column counts

## Binary Pre-Selection Flags

The binary flags in the first row serve as a **pre-filtering mechanism**, typically coming from a machine learning classifier (referenced in the original code as "Song's ML").

### Purpose
- **1**: Include this trajectory in analysis (e.g., high-quality signal, passed QC)
- **0**: Skip this trajectory (e.g., low SNR, failed QC, control sample)

### Typical Use Cases
1. **Quality Control**: ML model identified good vs. bad traces
2. **Sample Type**: Distinguish experimental vs. control samples
3. **Manual Curation**: User-defined selection of interesting trajectories
4. **Pre-screening**: Previous analysis identified candidates for detailed HMM analysis

## File Creation Tips

### Python Example
```python
import numpy as np
import pandas as pd

# Create sample data
n_trajectories = 10
n_timepoints = 100
binning = 0.1  # seconds

# Generate binary flags (1 = process, 0 = skip)
binaries = np.random.choice([0, 1], size=n_trajectories, p=[0.3, 0.7])

# Generate time points
time = np.arange(0, n_timepoints * binning, binning)

# Generate intensity data (example: 2-state photobleaching)
data = np.random.poisson(15, size=(n_timepoints, n_trajectories))
# Add bleaching step at midpoint
data[n_timepoints//2:, :] = np.random.poisson(7, size=(n_timepoints//2, n_trajectories))

# Create DataFrame
df = pd.DataFrame(data, index=time, columns=binaries)

# Save as whitespace-delimited file
df.to_csv('hmm_input.txt', sep=' ', float_format='%.1f')
```

### Expected Output Format
```
       1    0    1    1    0    1    1    0    1    1
0.0   15.0  14.0  16.0  13.0  15.0  14.0  17.0  15.0  14.0  16.0
0.1   16.0  15.0  14.0  15.0  16.0  15.0  15.0  14.0  15.0  15.0
...
5.0    7.0   8.0   6.0   7.0   8.0   7.0   6.0   8.0   7.0   7.0
...
```

## Validation Checklist

Before running the analysis, verify:

- [ ] File is **whitespace-delimited** (not comma-separated)
- [ ] First row contains only **0s and 1s** (binary flags)
- [ ] At least one column has **binary flag = 1**
- [ ] All subsequent rows contain **numeric values**
- [ ] No **missing values** (NaN or empty cells)
- [ ] All rows have the **same number of columns**
- [ ] File is saved with **UTF-8 encoding**
- [ ] Intensity values are **non-negative** (counts/photons)

## Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Comma-separated file | Parse error or wrong data structure | Re-save with spaces/tabs as delimiter |
| Missing binary row | All columns processed or index errors | Add binary flag row at top |
| Non-numeric values | Type conversion errors | Ensure all data cells are numeric |
| Mixed delimiters | Columns not aligned properly | Use consistent whitespace delimiter |
| Wrong encoding | Special characters or read errors | Save file as UTF-8 |

## Output Expectations

Given a properly formatted input file with:
- 100 time points (10 seconds at 0.1 s binning)
- 20 trajectories
- 15 trajectories with binary flag = 1

The analysis will produce:
- **Reconstructed_paths.xlsx**: State paths for 15 trajectories
- **Optimized_Rates.xlsx**: Emission rates, std devs, and stoichiometry histogram
- **Figures folder**: BIC curves and reconstruction plots for each trajectory
