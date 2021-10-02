# Social NLP S2DS Project: Social Metric Identifier

## Contributors

Lucas Das Dores, Isabel Ellerbrock, Jasper Hajonides, Angel Luis Jaso Tamame, Juliet Thompson

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

To return the identified pages and text for each of the metrics and all reports in the specified folder, 

```python
python generate_final_dataframe.py
```

More specifically, you can extract data from individual reports and for individual metrics using the `extract_given_metric.py` function.

``` python 
path = ‘/path/to/annual_report.pdf’
metric = ‘metric’ #e.g., ‘n_employees’, ‘ltifr’ 
df = extract_given_metric(path, metric)
```

The list of metrics that it is currently possible to locate within a PDF (if present) is as follows:

```python
“n_employees”
“n_contractors”
“n_fatalities”
“ltifr”
“trifr”
“international_diversity”
“company_gender_diversity”
“board_gender_diversity”
“company_ethnic_diversity”
“board_ethnic_diversity”
“healthcare”
“parental_care”
“ceo_pay_ratio”
```



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

