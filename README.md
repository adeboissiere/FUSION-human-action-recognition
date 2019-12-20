FUSION: Full Use of Skeleton and Infrared in Optimized Network for Human Action Recognition
==============================

We propose a novel deep network fusing skeleton and infrared data for Human Action Recognition. The network is tested on the largest RGB+D dataset to date, NTU RGB+D. We report state of the art performances with over 90% accuracy on both cross-subject and cross-view benchmarks. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├── missing_skeleton.txt   
    │   │   ├── samples_names.txt   
    │   │   ├── ir.h5   
    │   │   ├── ir_cropped.h5
    │   │   ├── ir_cropped_moving.h5 (optional)
    │   │   ├── ir_skeleton.h5
    │   │   ├── skeleton.h5   
    │   │      
    │   └── raw            <- The original, immutable data dump.
    │       ├── nturgb+d_ir         <- Raw IR videos (*.avi)
    │       ├── nturgb+d_skeletons  <- Raw skeleton files (*.skeleton)    
    │
    ├── docs               <- Sphinx generated documentation
    │
    ├── models             <- Trained models and paper results.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Create h5 datasets (see make data documentation)
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


Getting started
------------
The first step to replicate our results is to clone the project and create a virtual environment using the Makefile. After that, the raw data should be downloaded and placed according to the default Project Organization provided above. Then various h5 datasets will have to be created using the Makefile. This will take a while but is a more practical way of handling data. Once the h5 files are created, you are ready to replicate our results or use them to implement your own models!

The data used comes from the [NTU RGB+D dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

1. Clone project
    `git clone https://github.com/gnocchiflette/FUSION-human-action-recognition`

2. Create virtual environment 
    `make create_environment`

3. Activate environment (do so every time you work on this repository)
    `workon fusion` (for virtual env wrapper)
    `source activate fusion` (for conda)

4. Install requirements
    `make requirements`

5. Download raw data from the [NTU RGB+D website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp), decompress archives and place files as described in the Project Description above.

6. Run the `make data` commands to create the h5 files.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
