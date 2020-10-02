
**Author**: Tinh Son

**Last updated**: 2020-05-06

This script extracts COVID-19 cases within the US, maps the data available from each date, stitches the maps together. Finally, performs MCMC parameter estimation for beta and gamma in SIR model, then projects the infected prediction for the next 365 days from the lastest infected data, assuming an average state with population of 6 millions under no intervention measures. 

**Required packages before running script:**

_Essentials_:
- cv2 
- numpy
- pandas
- matplotlib(pyplot, [basemap](https://matplotlib.org/basemap/users/installing.html))
- seaborn
- cycler
- PIL

_Bookkeeping_:
- datetime
- os
- glob
- re
- itertools 
- gitdir
- subprocess

_Computational_:
- scipy
- pymcmcstat

_Extra for pandas to read excel files_:
```
pip install xlrd
```

**Basemap installation**

Windows: 
- Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Run conda console as administrator. Type: 
```
conda install basemap
conda install basemap-data-hires
python -m pip install Pillow
``` 
- This means that you will be using python via conda to run this script, so install other modules above under conda environment if necessary.

Linux, under terminal:

```
sudo apt-get install libgeos-3.5.0
sudo apt-get install libgeos-dev
sudo pip install https://github.com/matplotlib/basemap/archive/master.zip
```

**cv2 installation**
```
pip install opencv-python
```
_To run the script_:

```python Map_generation_projection_script.py```

_Outputs_:

- Map (Scatter heat) of COVID-19 confirmed and death cases within the continental US for each date, starting from the earliest available date to most recent (in folder labeled "Progress").

- Video of case progression (in current folder).

- MCMC Density graphs of variables beta and gamma in simple SIR model (in folder labled "MCMC").

- Final infected projection graph 365 days from the most recent data (in current folder). 


The script is designed to be a hands-off process, where user will only require the population by state spreadsheet from the [US Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html) (nst-est2019-01.xlsx), which is already included in this folder. 

COVID 19 Data is acquired automatically from the Johns Hopkins CSSEGIS on [Github](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports). Relevant spreadsheet attributes are: Date, Coordinate, Confirmed, Recovered, and Deaths. 

MCMC settings: Adaptive Metropolis samplers + delayed rejection ([DRAM](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=5&ved=2ahUKEwjwgoON_4vpAhWElHIEHa37DDgQFjAEegQIBBAB&url=https%3A%2F%2Fwiki.helsinki.fi%2Fdownload%2Fattachments%2F33885362%2FPreprint374.pdf&usg=AOvVaw2-hLWl4xT3DVTe_A5iKVab)), running 10000 simulations. 

It takes about **_30 minutes_** for the script to complete running.  
