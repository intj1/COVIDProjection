'''@Author: Tinh Son
Last updated: 2020-04-28
Script extracts COVID-19 cases within the US, maps the data available from each date, stitches the maps together. 
Finally, performs MCMC parameter estimation for beta and gamma in SIR model, 
then project the infected prediction within 365 days from the latest infected data, 
assuming an average state with population of 6 millions.'''

#Essentials
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from mpl_toolkits.basemap import Basemap
#gitdir is used to download the dataset from github
import gitdir
import subprocess
#Bookkeeping tools
import os
import glob
import re
from itertools import product as prod
from datetime import datetime as dt
#Image stitching module
import cv2
#Computational modules
from scipy.integrate import odeint
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.MCMC import MCMC
from pymcmcstat import mcmcplot as mcp

'''plot settings'''
sns.set_style("ticks")
sns.set_context(font_scale = 1, rc = {"lines.linewidth": 2.0, 'lines.markersize': 5})
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
mpl.rc('axes', prop_cycle = (cycler('color', ['r', 'k', 'b','g','y','m','c','gray']) ))
mpl.rc('text', usetex = False)
tw = 1.5
sns.set_style({"xtick.major.size": 6, "ytick.major.size": 6,
               "xtick.minor.size": 4, "ytick.minor.size": 4,
               'axes.labelsize': 24,
               'xtick.major.width': tw, 'xtick.minor.width': tw,
               'ytick.major.width': tw, 'ytick.minor.width': tw})
mpl.rc('xtick', labelsize = 18)
mpl.rc('ytick', labelsize = 18)
mpl.rc('axes', linewidth = 1.75)
sns.set_style({'axes.labelsize': 24, 'axes.titlesize': 24})

##################################################
class dfProcess():
    def __init__(self, path, country):
        self.__dates = sorted([re.search(r'(\d+-\d+-\d+)', d).group() for d in glob.glob(os.path.join(path, "*.csv"))])
        self.__df = {re.search(r'(\d+-\d+-\d+)',i).group(): pd.read_csv(i) for i in glob.glob(os.path.join(path, "*.csv"))}
        self.c_format = {'Country/Region', 'Country_Region'} #Upon looking at csv formats, I realized there is an inconsistency in keys. This should fix it.
        self.coords_format = {'Latitude', 'Longitude', 'Lat', 'Long_'}
        self.s_format = {'Province/State', 'Province_State'}
        self.__countries = sorted({country for d in self.__dates for country in self.__df[d][(self.c_format & set(self.__df[d].columns)).pop()]})
        
        #1
        self.__cDates = sorted({i for i in self.dates if country in set(self.df[i][(self.c_format & set(self.df[i].columns)).pop()])} &
                               {j for j in self.dates if len(set(self.df[j][(self.coords_format & set(self.df[j].columns))]))})

        #2
        self.__country = self.countryBreakDown(country)
        
        #3 
        self.__states = self.statesBreakDown()
        
    @property
    def dates(self):
        return self.__dates
    
    @property
    def df(self):
        return self.__df
    
    @property
    def countries(self):
        return self.__countries
    
    @property
    def country(self):
        return self.__country
    
    @property 
    def cDates(self):
        return self.__cDates
    
    @property
    def states(self):
        return self.__states
    
    def extractStats(self, date):
        return self.df[date].groupby((self.c_format & set(self.df[date].columns)).pop())[['Confirmed', 'Recovered', 'Deaths']].sum()

    def statsPerCountry(self, country):
        avail = sorted({i for i in self.dates if country in set(self.df[i][(self.c_format & set(self.df[i].columns)).pop()])})
        stats = pd.DataFrame(map(lambda x: np.array(self.extractStats(x).loc[country]), avail), 
                             columns = ['Confirmed', 'Recovered', 'Deaths'], 
                             index = avail)
        stats.index.name = 'Dates'
        return stats
    
    #Break down stats by State/Province
    def countryBreakDown(self, country):
        #Data processing is getting more difficult than I though. Let's hope that they stop changing the data format as the day goes on.
        df_df = {d: self.df[d][self.df[d][(self.c_format & set(self.df[d].columns)).pop()] == country][[(self.s_format & set(self.df[d].columns)).pop(), 
                                           sorted(self.coords_format & set(self.df[d].columns))[-1],
                                           sorted(self.coords_format & set(self.df[d].columns))[0], 
                                           'Confirmed', 'Recovered', 'Deaths']].set_index((self.s_format & set(self.df[d].columns)).pop()) for d in self.cDates}
        #Add multi-Index
        for k, v in df_df.items():
            coords = list(prod(['Coordinates'], v.columns[:2]))
            coords.extend(list(prod([k], v.columns[2:])))
            v.columns = pd.MultiIndex.from_tuples(coords)
            v.index.names = ['State/Province'] #Rename index level for consistency
        return df_df
        
       #Statesbreak down. PASS IN self.country
    def statesBreakDown(self):
        statesdict = {"D.C":"District of Columbia","AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
        DPfilter = {'Unassigned Location (From Diamond Princess)','Grand Princess Cruise Ship'} 
        #Use latest date as scheme for merge. This relies heavily on the latest date contains coords
        latest = self.cDates[-1]
        dff = self.country[latest].groupby(self.country[latest].index).sum()
        dff.drop(columns = dff.columns.levels[0][0], level = 0, inplace = True) #Drop the date columns, so all we have now are the states
        if len(dff.columns.levels) == 2:
            dff.drop(columns = dff.columns.levels[0][1], level = 0, inplace = True) #DROPPING COORDINATES TOO. COMMENT IF NEEDED
        #####################################################################
        for i in self.cDates:
            df = self.country[i].groupby(self.country[i].index).sum()
            #Some dates do not have the coordinates column, Need to check length of level 0 multiIndex
            if len(df.columns.levels) == 2:
                df.drop(columns = df.columns.levels[0][1], level = 0, inplace = True)
            #before merging, filter state/province columns from previous dates
            if len(set(df.index) & DPfilter):
                df.drop(index = list(set(df.index) & DPfilter), inplace = True)
            filtered = [re.findall(r'([A-Z]{2}|[A\.-Z]{3})', x)[0] for x in df.index if len(re.findall(r'([A-Z]{2}|[A\.-Z]{3})', x)) == 1] 
            if len(set(statesdict) & set(filtered)):
                df.index = [statesdict[s] for s in filtered]
                df = df.groupby(df.index).sum()
                df.index.names = ['State/Province']
            dff = pd.merge(dff, df, how = 'left', on = 'State/Province')
        #Depending on new locations, might need to add or remove from below
        dff.drop(index = ['Recovered','Virgin Islands', 'Northern Mariana Islands', 'Grand Princess', 'Diamond Princess', 'Puerto Rico'], inplace = True)
        dff.fillna(0, inplace = True) #Fill na for states whose values don't exist from previous dates
        return dff 
    
    def getStats(self, state):
        s = pd.DataFrame(self.states.loc[state]).transpose()
        c = list(map(lambda x: s[x]['Confirmed'].values[0], self.cDates))
        r = list(map(lambda x: s[x]['Recovered'].values[0], self.cDates))
        d = list(map(lambda x: s[x]['Deaths'].values[0], self.cDates))
        return (c, r, d)
    
    #Graphs stats for states: REQUIRES dff
    def statesGraph(self, state):
        fig, ax = plt.subplots(1, figsize = (15, 12))
        c, r, d = self.getStats(state)
        ax.plot(self.cDates, c, label = 'Confirmed')
        ax.plot(self.cDates, r, label = 'Recovered')
        ax.plot(self.cDates, d, label = 'Deaths')
        ax.set_xticks(self.cDates[::5])
        ax.set_xticklabels(self.cDates[::5], rotation = 45)
        ax.legend()
        plt.title(state)
        
    #Valid columns are: 'Confirmed', 'Recovered', 'Deaths'
    def barGraphPerCountry(self, c_frame, column):
        fig, ax = plt.subplots(figsize = (12, 6))    
        fig = sns.barplot(x = list(c_frame.index), 
                          y = column, 
                          data = c_frame,
                          orient = 'v')
        ax.set_xticklabels([])
        
    def mapPlot(self, c_frame, column, save = 1):
        date = c_frame.columns.levels[0][0]
        cf = c_frame.copy(deep = True)
        cf.columns = cf.columns.droplevel(0)
        #Filter coordinates from US cases outside mainland (excluding Alaska)
        cf = cf[(cf[cf.columns[0]].between(-130, -55) & cf[cf.columns[1]].between(26, 50))]
        
        
        '''Bookkeeping'''
        lon, lat = cf[cf.columns[0]].values, cf[cf.columns[1]].values
        confirmed, color = cf['Confirmed'].values, cf[column].values 
        max_col = self.country[self.cDates[-1]][self.cDates[-1]][column].values
        
        '''Draw map'''
        fig = plt.figure(figsize = (20, 20))
        m = Basemap(projection = 'lcc', resolution = 'l',
            width = 5E6, height = 4E6, 
            lat_1 = 26., lat_2 = 50, lat_0 = 35, lon_0 = -98)
        m.shadedrelief()
        m.drawcoastlines(color = 'gray')
        m.drawcountries(color = 'black')
        m.drawstates(color = 'gray')
        
        '''Scatter'''
        cNorm = plt.Normalize(vmin = 0, vmax = max_col.max())
        m.scatter(x = lon, y = lat, latlon = True,
                  s = confirmed,
                  c = color, cmap = plt.get_cmap('jet'),
                  norm = cNorm,
                  alpha = 0.5)
    
        cbar = plt.colorbar(label = column, shrink = 0.60)    
        
        plt.ylabel('Latitude', fontsize = 14)
        plt.xlabel('Longitude', fontsize = 14)
        plt.title('Confirmed Cases for {}: {}'.format(date, int(confirmed.sum())), fontsize = 20)
        
        #Size legend
        for a in [10, 100, 500]:
            plt.scatter([], [], c = 'k', alpha = 0.9, s = a,
                        label = str(a) + ' Confirmed')
        plt.legend(loc = 3, fontsize = 16)
        
        if not save:
            plt.show()
            
        else:
            if not os.path.exists('Progress'):
                os.makedirs('Progress')
            plt.savefig("Progress/{}.png".format(date), bbox_inches = 'tight', dpi = 100)
            plt.clf()
            plt.close()
            
class SIR:
    def __init__(self, t, delay): #parameter values declared when run model
        self.__beta = 1
        self.__gamma = 1
        self.__R = self.__beta/self.__gamma 
        self.__N = 1
        ##Delay value for Susceptible to be included in the model
        self.__steps = []
        self.__t = t
        self.__delay = delay
        
    @property
    def beta(self):
        return self.__beta
    @beta.setter
    def beta(self, val):
        self.__beta = val
        self.__R = self.beta/self.gamma 

    @property
    def gamma(self):
        return self.__gamma
    @gamma.setter
    def gamma(self, val):
        self.__gamma = val
        self.__R = self.beta/self.gamma 
        
    @property
    def N(self):
        return self.__N
    @N.setter
    def N(self, val):
        self.__N = val
        
    @property
    def steps(self):
        return self.__steps
    @steps.setter
    def steps(self, val):
        self.__steps = val
    
    @property 
    def t(self):
        return self.__t
    @t.setter
    def t(self, val):
        self.__t = val
        
    @property 
    def delay(self):
        return self.__delay
    @delay.setter
    def delay(self, val):
        self.__delay = val
        
    def __model(self, z, t, beta, gamma): 
        S, I, R = z[0], z[1], z[2]
        #Susceptible
        SDot = -(beta * (self.steps[(-1 - self.delay)%len(self.steps)][0] * self.steps[(-1 - self.delay)%len(self.steps)][1]))/self.N
        #Infected
        IDot = (beta * (S * I))/self.N - gamma * I 
        #Removed
        RDot = gamma * I
        #Dead
        dzdt = [SDot, IDot, RDot]
        return dzdt
                
    def runmodel(self, beta, gamma, N, S, I, R):
        self.beta, self.gamma, self.N = beta, gamma, N
        #initial conditions
        self.steps = []
        z0 = [S, I, R]
        self.steps.append(z0) #access dfvious calculation due to time delay
        
        sr = np.empty_like(self.t)
        ir = np.empty_like(self.t) 
        rr = np.empty_like(self.t)
        sr[0], ir[0], rr[0] = z0[0], z0[1], z0[2]
        
        #Integrate 
        for i in range(1, len(self.t), 1):
            tspan = [self.t[i -1], self.t[i]]
            z = odeint(self.__model, z0, tspan, args = (self.beta, self.gamma))
            sr[i] = z[1][0]
            ir[i] = z[1][1]
            rr[i] = z[1][2]
            z0 = z[1]
            #save values for DDE
            self.steps.append(z[1])
            
        results = np.array([sr, ir, rr])
        results[results < 0] = 0
        results[results > self.N] = self.N
        return results

    

def main():
    #############################################################
    print('Getting data from github...')
    try:
        subprocess.run(['gitdir', 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports/'])
        directory = './csse_covid_19_data/csse_covid_19_daily_reports'
        df = dfProcess(directory, 'US')
    except:
        print('Directory error.')
        
    print('Countries:\n {} \n'.format(df.countries))
    print('Dates in database:\n {} \n'.format(df.dates))
    print('Data for US available from the following dates:\n {} \n'.format(df.cDates))
    print('Data for US States Isolated: \n {}'.format(df.states))

    ############################################################

    '''Map Creation'''
    #Will take a while 
    #Create maps for all dates
    print('\n Creating maps for each available date...')
    mpl.use('Agg')
    for i, date in enumerate(df.country.keys()):
        print("Creating map for {}.........{}/{}".format(date, i + 1, len(df.country.keys())))
        df.mapPlot(df.country[date], 'Deaths', 1)
    print('Maps are saved in Progress folder...')
    ############################################################
    '''Stitching images together'''
    print('Creating video of maps...')
    path = sorted(glob.glob(os.path.join('./Progress/*.png')))
    files = [cv2.imread(file) for file in path]
    height, width, layers = files[0].shape
    size = (width, height)
    files = [cv2.resize(file, size) for file in files]

    out = cv2.VideoWriter('ConfirmedvsDeaths.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, size) 
    for file in files:
        for _ in range(15):
            out.write(file)
    out.release()
    print('Done.')
    #########################################################
    '''Parameter estimation'''
    #Define cost function for MCMC
    def ssfunc(q, data):
        y = data.ydata[0]
        beta, gamma, N, S, I, R = q
        ymodel = model.runmodel(beta, gamma, N, S, I, R)
        res = ymodel[1] - y
        return (res**2).sum()

    print('Setting up auxilary files and MCMC parameters')
    #Aux data
    census = pd.read_excel('nst-est2019-01.xlsx', encoding = 'UTF-8')
    census['State'] = list(map(lambda x: re.sub(r'\.', '', x), census['State'])) #Some states has a period in the beginning, cleaning that
    census.set_index('State', inplace = True)

    #Scatter
    confirmed = [j for k in df.states.columns.levels[0][:-2] for j in df.states[k]['Confirmed']]
    timespan = np.linspace(0, len(df.cDates), len(confirmed))
    model = SIR(timespan, 14)

    #invoke module
    mcmcstat = MCMC()
    mcdata = DataStructure() #Just doing this for the sake of following proper format
    mcmcstat.data.add_data_set(x = timespan, y = confirmed)

    #apply cost function
    mcmcstat.model_settings.define_model_settings(sos_function = ssfunc)

    #simulation options. Using DRAM method
    mcmcstat.simulation_options.define_simulation_options(
        nsimu = 10.0e3, 
        updatesigma = True,
        method = 'dram',
        adaptint = 100,
        verbosity = 1,
        waitbar = 1)

    #Parameters of interest
    mcmcstat.parameters.add_model_parameter(
        name = 'beta',
        theta0 = 0.01,
        minimum = 0.00001,
        maximum = 0.5)

    mcmcstat.parameters.add_model_parameter(
        name = 'gamma',
        theta0 = 0.01,
        minimum = 0.001,
        maximum = 0.5/2.1)

    '''Population and initial conditions, these are fixed.'''
    N, S, I = census['Population'].sum()/len(df.states), census['Population'].sum()/len(df.states) - 1, 1
    R = 0
    mcmcstat.parameters.add_model_parameter(
        name = 'N',
        theta0 = N,
        sample = False)

    mcmcstat.parameters.add_model_parameter(
        name = 'S',
        theta0 = S,
        sample = False)

    mcmcstat.parameters.add_model_parameter(
        name = 'I',
        theta0 = I,
        sample = False)

    mcmcstat.parameters.add_model_parameter(
        name = 'R',
        theta0 = R,
        sample = False)

    print('Running MCMC. Will take a while...')
    mcmcstat.run_simulation()
    print('Done.')

    ####MCMC outputs
    mcmcresults = mcmcstat.simulation_results.results
    burnin = int(mcmcresults['nsimu']/2)

    #Statistics
    chain = mcmcresults['chain']
    s2chain = mcmcresults['s2chain']
    sschain = mcmcresults['sschain']
    names = mcmcresults['names']
    mcmcstat.chainstats(chain[burnin:,:], mcmcresults)
    print('Acceptance rate: {:6.4}%'.format(100*(1 - mcmcresults['total_rejected'])))
    print('Model Evaluations: {}'.format(mcmcresults['nsimu'] - mcmcresults['iacce'][0] + mcmcresults['nsimu']))

    if not os.path.exists('MCMC'):
        os.makedirs('MCMC')
    fig = mcp.plot_density_panel(chain[burnin:,:], names)
    fig.savefig("MCMC/Density.png", bbox_inches = 'tight', dpi = 100)
    fig = mcp.plot_pairwise_correlation_panel(chain[burnin:,:], names)
    fig.savefig("MCMC/Pairwise.png", bbox_inches = 'tight', dpi = 100)
    print('MCMC parameter plots are saved in MCMC')
    ##########################################################
    print('Creating final projection graph...')

    #difference from first date to last day
    ddiff = abs((dt.strptime(df.cDates[0], '%m-%d-%Y') - dt.strptime(df.cDates[-1], '%m-%d-%Y')).days)
    #projected parameters
    pbeta, pgamma = mcmcresults['mean'][0], mcmcresults['mean'][1]
    #Projecting 360 days, fingers crossed
    ptimespan = np.linspace(ddiff, ddiff + 365, 365 * len(df.states))
    model = SIR(ptimespan, 14)
    #define current susceptible, infected, removed.
    cconfirmed = df.statsPerCountry('US').loc[df.cDates[-1]]['Confirmed']/len(df.states)
    cremoved = (df.statsPerCountry('US').loc[df.cDates[-1]]['Recovered'] + df.statsPerCountry('US').loc[df.cDates[-1]]['Deaths']) / len(df.states)
    csus = S - cconfirmed - cremoved


    projection = model.runmodel(pbeta, pgamma, N, csus, cconfirmed, cremoved)
    pfig, pax = plt.subplots(1, 1, figsize = (35, 8))
    pax.plot(ptimespan, projection[1])
    pax.set(title = 'Projected Infectives', xlabel = 'Days since 03-01-2020')
    plt.savefig('./FinalProjection.png', bbox_inches = 'tight', dpi = 100)

    print('FinalProjection graph saved in current folder...Done')

if __name__ == '__main__':
    main()