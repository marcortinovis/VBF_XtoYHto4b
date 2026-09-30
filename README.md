# VBF_XtoYHto4b
DESY 2026 Summer Student Project on VBF production of XtoYHto4b




**Requirements:**

All the scripts are in python and all the used libraries are listed in requirements.txt.

ROOT was used as pyROOT, that is directly in python. ```root-config --version: 6.40.04```

```paths.py``` needs to be provided (see below).

**Description:**

- The first part of the project (individual trigger efficiencies and combinations) can be reproduced with the 
  first block of programs (until make_plots.py)
	- the two cycle scripts are used to compute efficiencies, written into csvs. 
	- then make_plots.py can be called to plot everything, with the desired set of options.
- The rest of the project is based on the second block
	- differential efficiencies, cut-flow, vbf pair selection, purity evaluation and DNN input feature 
	  computation are all produced by running the script "fetch_and_analyse.py", in particular the first 
	  time with the --mjj option. Then plots can be made with the five plotting scripts 
	  (make_bjets_plots.py, make_cut_flow_plots.py, make_distr_plots.py, make_purity_plots.py, 
	  make_vars_plots.py) each with their specific options, but again always with the --mjj option.
	- Only after that, the DNN can be trained. The first step is to prepare the dataset with 
	  dataset_preparation.py, based on the features already computed by fetch_and_analyse. Then, the 
	  model can be trained with train.py, which calls on its own the plotting functions. Then the shap 
	  explanation can be performed with shap_expl.py.
	- After the DNN is ready, "fetch_and_analyse.py" can be run again with the --dnn option in order to 
	  get all the information needed.

**Structure:**
```text
.   # -------------------------------------------------------------------------------------------------------
│
├── requirements.txt                            - list of all the python packages installed 
│                                                 and used during the project
│
├── paths.py					                          - list of paths where all the samples are stored
│						                                      (to be provided)
│
│   # -------------------------------------------------------------------------------------------------------
│
├── compare_csvs.py                             - compare the signal selection efficiencies.
│                                                 needs cycle_id_bkgs_read_NanoAOD_trigger.py and
│                                                 cycle_id_bkgs_read_NanoAOD_trigger.py to be run beforehand
├── csvs/
├── cycle_id_bkgs_read_NanoAOD_trigger.py       - compute signal selection efficiencies for bkg samples
├── cycle_id_sigs_read_NanoAOD_trigger.py       - compute signal selection efficiencies for signal samples
├── imgs/
├── make_plots.py                               - make all the efficiency plots
│
│
│   # -------------------------------------------------------------------------------------------------------
│
│
├── fetch_and_analyse.py                        - (!) main script. collect data from the remote location, 
│                                                 compute differential efficiencies, 
│                                                 checks cut-flow conditions, performs vbf pair selection,
│						  checks purity and compute the input nn variables
├── var_funcs.py                                - definitions of functions to compute the nn features
├── make_bjets_plots.py                         - make plots relative to the 4 b jets system. 
│                                                 fetch_and_analyse.py needs to be run beforehand.
├── make_cut_flow_plots.py                      - make cut-flow plots.
│                                                 fetch_and_analyse.py needs to be run beforehand.
├── make_distr_plots.py                         - make differential efficiencies plots.
│                                                 fetch_and_analyse.py needs to be run beforehand.
├── make_purity_plots.py                        - make purity plots.
│                                                 fetch_and_analyse.py needs to be run beforehand.
├── make_vars_plots.py                          - make nn feature distributions plots.
│                                                 fetch_and_analyse.py needs to be run beforehand.
│
│
│
├── mjj_sel                                     - fetch_and_analyse.py results for --mjj
│   ├── bjets/
│   ├── csvs/
│   ├── npzs/
│   ├── plots/
│   ├── vbf_evt_ids/
│   └── vbf_parquets/
│
│
│
├── dnn                                         - neural network
│   ├── datasets
│   │   ├── dataset_preparation.py              - script for data preparation.
│   │   │                                         needs fetch_and_analyse.py to be run with --mjj.
│   │   ├── datasets_alpha/
│   │   └── datasets_beta/
│   ├── models
│   │   └── dnn.py                              - module to define the NNs
│   ├── plot_metrics.py                         - plot functions for metrics and performance
│   ├── results/
│   ├── shap_expl.py                            - script for shap explanation. 
│   │                                             needs model to be trained, uses files in results/
│   └── train.py                                - script for training, needs dataset_preparation.py 
│                                                 to be run for the target dataset
│
│
├── dnn_sel                                     - fetch_and_analyse.py results for --dnn
│   ├── csvs/
│   ├── dnn_eff/
│   ├── npzs/
│   ├── plots/
│   ├── vbf_evt_ids/
│   └── vbf_purity_parquets/
│
│
│
└── mjj_dnn_comp                                - comparisons between fetch_and_analyse.py results for
    │                                             --mjj and --dnn. comes from make_purity_plots.py with
    │						  					  the --comp option
    └── plots/
```
