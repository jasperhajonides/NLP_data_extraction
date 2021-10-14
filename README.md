# Natural Language Processing Data Extraction from Business Reports

## Contributors

Jasper Hajonides in collaboration with: Lucas Das Dores, Isabel Ellerbrock, Angel Luis Jaso Tamame, Juliet Thompson

### Set up

1. Create a virtual environment called "SocialNLP" with `conda env create -n SocialNLP -f SocialNLP_envrionment.yml`

2. run `source activate SocialNLP` 

Alternative:
`pip install -r requirements.txt`


### Maintenance
If you install new packages, add these to the `requirements.txt` file like this (after installing pip-chill) :
```
pip chill > requirements.txt
```

## General function description

The functions we wrote iteratively read in pdf’s from a specified directory and perform NLP and machine learning to return the most likely page at which the specified metric occurs. 




## The directory structure of the project is as follows: 

```
├── README.md          
├── additional_information.md          
├── data
│   ├── external                <- csv files with sector and industry information for given companies
│   ├── processed               <- Training datasets and final_dataframe outcomes for each metric.
│   └── raw                     <- Raw PDF files.
│
├── models                      <- Saved models for each metric in .pkl format
│
├── scripts                     <- Scripts for project functionality
│   └── archive                 <- Archived avenues of exploration.
│
├── notebooks                   <- Jupyter notebooks containing example processing steps.
│   └── archive                 <- Archived avenues of exploration.
│
├── reports           
│   └── figures                 <- Generated graphics and figures
│
├── SocialNLP_environment.yml   <- The requirements file for reproducing the conda environment
|
├── requirements.txt            <- The requirements file for installation with pip`
```

