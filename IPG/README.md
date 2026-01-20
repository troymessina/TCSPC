# File Descriptions
* `tcspc_analysis.py` - the main analysis routines for global regularization
* `tcspc_corrected_workflow.py`-  corrected workflow to include IRF selection, data truncation, etc.
* `GRIP_test.ipynb` - an example of usage that uses data in the the `data` and `irf` folders

The `unused_irfs` can be copied into the `irf` folder to see instrument response selection. The data has been normalized to a maximum of 1, and the IRFs are integrated to 1. Neither normalization should be necessary. However, better fits are usually obtained with normalized IRFs and when the IRF background is removed.
