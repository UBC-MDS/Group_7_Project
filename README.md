# Bank Marketing Analysis

A Bank Telemarketing Success Detection Framework.

## Usage

### Environment Setup

1. Setup your Python environment: e.g., Miniconda Python 3.11 [[Guide]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

2. Clone the repository:
```
$ git clone https://github.com/UBC-MDS/Group_7_Project.git
```

3. Install virtual environment 
```
$ conda env create -f environment.yml
```

4. Activate the virtual environment
```
$ conda activate 522_project_env
```

5. Launch Jupyter Lab to open the `analysis.ipynb`
```
$ jupyter lab
```

### Data Download
Our data is part of the UC Irvine Machine Learning Repository. It can be downloaded using the `ucimlrepo` library.
1. Install the library (if you create the virtual environment using the `environment.yml` file, then the library is already installed)
```
$ pip install ucimlrepo
```

2. Import the fetch method from library
```
from ucimlrepo import fetch_ucirepo
```

3. Fetch the data set by ID
```
bank_marketing = fetch_ucirepo(id=222)
```


### Data Description

In this project, we utilized a dataset concerning direct marketing campaigns conducted by a Portuguese banking institution, as provided by Sérgio Moro, P. Rita, and P. Cortez in 2012 (Moro, S., Rita, P., and Cortez, P.). The dataset was sourced from UC Irvine's Machine Learning Repository and can be accessed via the following link: https://archive.ics.uci.edu/dataset/222/bank+marketing. Comprising 16 features and 45,211 instances, each row of the dataset corresponds to information about an individual client of the Portuguese bank. The primary objective of the dataset creators was to predict whether a client would subscribe to a term deposit, a target variable indicated by the 'y' column. In our analysis, we also utilized this column as our target variable.

The columns of this data are defined as below:
| Variable Name | Role  | Type | Demographic| Description| Units | Missing Values |
|:----:|:--------:|:---------:|:----------:|:------:|:------:|:------:|
| age | Feature | Integer |Age  | | | no |
| job | Feature | Categorical | Occupation  | type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown') | | no |
| marital | Feature | Categorical | Marital Status  | marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed) | | no |
| education | Feature | Categorical | Education Level  | (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown') | | no |
| default | Feature | Binary |   | has credit in default? | | no |
| balance | Feature | Integer |  | average yearly balance| euros| no |
| housing | Feature | Binary |  | has housing loan? | | no |
| loan | Feature | Binary |  | has personal loan?| | no |
| contact | Feature | Categorical |  |contact communication type (categorical: 'cellular','telephone') | | yes |
| day_of_week | Feature | Date |  | last contact day of the week | | no |
| month | Feature | Date |   | last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') | | no |
| duration | Feature | Integer |  | last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model. | | no |
| campaign | Feature | Integer | | number of contacts performed during this campaign and for this client (numeric, includes last contact) | | no |
| pdays | Feature | Integer | | number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted) | | yes |
| previous | Feature | Integer | |number of contacts performed before this campaign and for this client | | no |
| poutcome | Feature | Categorical | |outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success') | | yes |
| y | Target | Binary |  | has the client subscribed a term deposit? | | no |

## Result Discussion

Our analysis commenced with the retrieval of data from the repository. Following exploratory data analysis, we opted to exclude the `poutcome` and `contact` features from our dataset due to a considerable number of `NaN` values in these columns, which limited their utility in model training and predictions. Visualizing histograms of the features, categorized by class (subscription status), indicated significant differences in distributions, affirming our decision to include all other features in our model training. Additionally, we observed a notable class imbalance in our target variable. Consequently, we chose not to employ accuracy as the evaluation metric for our model, recognizing its inadequacy in assessing performance. Instead, we opted for the F2 score.

## Citation

If you find this code useful, please cite the original paper:

- Bauer, C. L., &amp; Miglautsch, J. (1992). A conceptual definition of direct marketing. Journal of Direct Marketing, 6(2), 7–17. https://doi.org/10.1002/dir.4000060204 
- Harris, C.R. et al., 2020. Array programming with NumPy. Nature, 585, pp.357–362.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
- McKinney, Wes. 2010. “Data Structures for Statistical Computing in Python.” In Proceedings of the 9th Python in Science Conference, edited by Stéfan van der Walt and Jarrod Millman, 51–56.
- Moro, S., Cortez, P., &amp; Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001 
- Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. 
- Nowak, G. J., &amp; Phelps, J. (1995). Direct marketing and the use of individual-level consumer information: Determining how and when “privacy” matters. Journal of Direct Marketing, 9(3), 46–60. https://doi.org/10.1002/dir.4000090307
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- VanderPlas, J. et al., 2018. Altair: Interactive statistical visualizations for python. Journal of open source software, 3(32), p.1057.
- Van Rossum, Guido, and Fred L. Drake. 2009. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.