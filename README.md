# Equipo35_MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Final project of Team 35

## Project Description

This project aims to implement an MLOps workflow for managing and deploying machine learning models. We use tools like DVC, MLFlow, and Minio to ensure reproducibility and efficiency in data and model management.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_team35 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_team35   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_team35 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- DVC
- MLFlow
- Minio

### Installation Steps

1. Clone the repository:
   ```powershell
   git clone https://github.com/cpenalozaj/Equipo35_MLOps.git
   cd Equipo35_MLOps
   

# Installation

```bash
# Create a virtual environment named 'equipo35_mlops'
python -m venv equipo35_mlops

# Activate the virtual environment
source equipo35_mlops/bin/activate
# for windows
<name>\Scripts\activate


# Install dependencies
python -m pip install -r requirements


# if any dependency missing
# add to requirements.txt and re run install
```

# DVC
### 1. Pull data
Enter your credentials, Access Key ID and Secret Access Key, in the file `~/.aws/credentials` or by using the command:
```bash
aws configure
```

Insert your credentials
```bash
❯ aws configure
AWS Access Key ID [****************]: <your access key id>
AWS Secret Access Key [****************]: <your secret access key>
Default region name [us-east-1]:
Default output format [None]:
```


Then, pull the data using the command:
```bash
dvc pull
```

### 2. Run DAG

```bash
dvc repro
```

### Contribution
Contributions are welcome! Follow these steps to contribute:

Create a new branch:

powershell

git checkout -b your-branch-name
Make your changes and commit them:

powershell

git add .
git commit -m "Description of your changes"
Push your branch to the remote repository:

powershell

git push origin your-branch-name
Create a Pull Request on GitHub.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or suggestions, you can contact us via email.