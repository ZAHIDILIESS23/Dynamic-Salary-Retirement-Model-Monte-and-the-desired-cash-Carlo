#!/usr/bin/env python
# coding: utf-8

# # Retirement Model
# 
# This is a retirement model which models salary with both a constant growth rate for cost of living raises as well as regular salary increases for promotions. The model is broken up into the following sections:
# 
# - [**Setup**](#Setup): Runs any imports and other setup
# - [**Inputs**](#Inputs): Defines the inputs for the model
# - [**Salaries**](#Salaries): Determining the salary in each year, considering cost of living raises and promotions
# - [**Wealths**](#Wealths): Determining the wealth in each year, considering a constant savings rate and investment rate
# - [**Retirement**](#Retirement): Determines years to retirement from the wealths over time, the main output from the model.
# - [**Results Summary**](#Results-Summary): Summarize the results with some visualizations
# - [**Sensitivity Analysis**](#Sensitivity-Analysis): Determine how sensitive the years to retirement is to the model inputs
# - [**Scenario Analysis**](#Scenario-Analysis): Determine how sensitive the years to retirement is to the model inputs
# -[**Adding unternal randomess to a python**](#internal_random)
# -[**Monte Carlo Simulation**](#Monte_Cartlo_Simulation)

# ## Setup
# 
# Setup for the later calculations are here. The necessary packages are imported.

# In[1]:


from dataclasses import dataclass
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import statsmodels.api as sm
import numpy_financial as npf


# ## Inputs
# 
# All of the inputs for the model are defined here. A class is constructed to manage the data, and an instance of the class containing the default inputs is created.

# In[15]:


@dataclass
class ModelInputs:
    starting_salary: int = 60000
    promos_every_n_years: int = 5
    cost_of_living_raise: float = 0.02
    promo_raise: float = 0.15
    savings_rate: float = 0.25
    interest_rate: float = 0.05
    years_to_retirement:int=24
    cash_spend_during_retirement:float=70000
        
model_data = ModelInputs()
data=model_data
data


# ## Salaries
# 
# Here the salary for each year is calculated. We assume that the salary grows at a constant rate each year for cost of living raises, and then also every number of years, the salary increases by a further percentage due to a promotion or switching jobs. Based on this assumption, the salary would evolve over time with the following equation:
# 
# $$s_t = s_0 (1 + r_{cl})^n (1 + r_p)^p$$
# 
# Where:
# - $s_t$: Salary at year $t$
# - $s_0$: Starting salary (year 0)
# - $r_{cl}$: Annual cost of living raise
# - $r_p$: Promotion raise
# - $p$: Number of promotions
# 
# And in Python format:

# In[16]:


def salary_at_year(data: ModelInputs, year):
    """
    Gets the salary at a given year from the start of the model based on cost of living raises and regular promotions.
    """
    # Every n years we have a promotion, so dividing the years and taking out the decimals gets the number of promotions
    num_promos = int(year / data.promos_every_n_years)
    
    # This is the formula above implemented in Python
    salary_t = data.starting_salary * (1 + data.cost_of_living_raise) ** year * (1 + data.promo_raise) ** num_promos
    return salary_t


# That function will get the salary at a given year, so to get all the salaries we just run it on each year. But we will not know how many years to run as we should run it until the individual is able to retire. So we are just showing the first few salaries for now and will later use this function in the [Wealths](#Wealths) section of the model.

# In[17]:


for i in range(6):
    year = i + 1
    salary = salary_at_year(model_data, year)
    print(f'The salary at year {year} is ${salary:,.0f}.')


# As expected, with the default inputs, the salary is increasing at 2% per year. Then at year 5, there is a promotion so there is a larger increase in salary.

# ## Wealths
# 
# The wealths portion of the model is concerned with applying the savings rate to the earned salary to calculate the cash saved, accumulating the cash saved over time, and applying the investment rate to the accumulated wealth.
# 
# To calculate cash saved, it is simply:
# 
# $$c_t = s_t * r_s$$
# 
# Where:
# - $c_t$: Cash saved during year $t$
# - $r_s$: Savings rate

# In[18]:


def cash_saved_during_year(data: ModelInputs, year):
    """
    Calculated the cash saved within a given year, by first calculating the salary at that year then applying the 
    savings rate.
    """
    salary = salary_at_year(data, year)
    cash_saved = salary * data.savings_rate
    return cash_saved


# To get the wealth at each year, it is just applying the investment return to last year's wealth, then adding this year's cash saved:
# 
# $$w_t = w_{t-1} (1 + r_i) + c_t$$
# Where:
# - $w_t$: Wealth at year $t$
# - $r_i$: Investment rate

# In[19]:


def wealth_at_year(data: ModelInputs, year, prior_wealth):
    """
    Calculate the accumulated wealth for a given year, based on previous wealth, the investment rate,
    and cash saved during the year.
    """
    cash_saved = cash_saved_during_year(data, year)
    wealth = prior_wealth * (1 + data.interest_rate) + cash_saved
    return wealth


# Again, just like in the [Salaries](#Salaries) section, we can now get the output for each year, but we don't know ultimately how many years we will have to run it. That will be determined in the [Retirement](#Retirement) section. So for now, just show the first few years of wealth accumulation:

# In[20]:


prior_wealth = 0  # starting with no cash saved
for i in range(6):
    year = i + 1
    wealth = wealth_at_year(model_data, year, prior_wealth)
    print(f'The wealth at year {year} is ${wealth:,.0f}.')
    
    # Set next year's prior wealth to this year's wealth
    prior_wealth = wealth


# With default inputs, the wealth is going up by approximately 25% of the salary each year, plus a bit more for investment. Then in year 6 we see a substantially larger increase because the salary is substantially larger due to the promotion. So everything is looking correct.

# ## Retirement
# 
# This section of the model puts everything together to produce the final output of years to retirement. It uses the logic to get the wealths at each year, which in turn uses the logic to the get salary at each year. The wealth at each year is tracked over time until it hits the desired cash. Once the wealth hits the desired cash, the individual is able to retire so that year is returned as the years to retirement.

# ## Calculate desired cash

# In[21]:


def get_desired_cash(data):
    return npf.pv(data.interest_rate,data.years_to_retirement,-data.cash_spend_during_retirement)


# ## Calculate years to retirement
# 

# In[22]:


def years_to_retirement(data: ModelInputs):
    
    # starting with no cash saved
    prior_wealth = 0  
    wealth = 0
    
    year = 0  # will become 1 on first loop
    desired_cash=get_desired_cash(model_data)
    
    print('Wealths over time:') # \n makes a blank line in the output.
    while wealth < desired_cash:
        year = year + 1
        wealth = wealth_at_year(model_data, year, prior_wealth)
        print(f'The wealth at year {year} is ${wealth:,.0f}.')

        # Set next year's prior wealth to this year's wealth
        prior_wealth = wealth
        
    # Now we have exited the while loop, so wealth must be >= desired_cash. Whatever last year was set
    # is the years to retirement.
    print(f'\nRetirement:\nIt will take {year} years to retire.')  # \n makes a blank line in the output.
    return year


# In[ ]:





# With the default inputs:

# In[23]:


years = years_to_retirement(model_data)


# # Sensitivity Analysis

# In[30]:


get_ipython().system(' pip install --upgrade sensitivity')


# In[24]:


from sensitivity import SensitivityAnalyzer


# In[25]:


def years_to_retirement_seperate(**kwargs):
    data=ModelInputs(**kwargs)
    return years_to_retirement(data)


# In[26]:


sensitivity_value={
    'starting_salary':[i*1000 for i in range(4,8)],
    'promos_every_n_years':[i for i in range(4,8)],
    'cost_of_living_raise':[i/100 for i in range(1,4)],
    'promo_raise':[i/100 for i in range(10,25,5)],
    'savings_rate':[i/100 for i in range(10,40,10)],
    'interest_rate':[i/100 for i in range(3,7)],
    }


# In[27]:


SensitivityAnalyzer(sensitivity_value,years_to_retirement_seperate)


# # Scenario-Analysis

# In[28]:


good_economy_data=ModelInputs(
    starting_salary=70000,
    promos_every_n_years=4,
    cost_of_living_raise=0.02,
    promo_raise=0.2,
    savings_rate=0.35,
    interest_rate=0.06)
bad_economy_data=ModelInputs(
    starting_salary=40000,
    promos_every_n_years=8,
    cost_of_living_raise=0.01,
    promo_raise=0.07,
    savings_rate=0.15,
    interest_rate=0.03)
cases={
    'Recession':bad_economy_data,
    'Normal':ModelInputs,
    'Expansion':good_economy_data
}
proba_scena={
    'Recession':0.2,
    'Normal':0.5,
    'Expansion':0.3
}
    


# In[29]:


for case_type,case_input in cases.items():
    ytr=years_to_retirement(case_input)
    weight_ytr=ytr*proba_scena[case_type]
    print(f' It would take {ytr} years to retirement in a {case_type} economy') 


# # internal_random

# In[30]:


import random
class ModelInputs:
    recession_prob:float=0.2
    expansion_prob:float=0.3
    case_name:tuple=('Recession','Normal','Expansion')
    starting_salary:tuple=[40000,60000,70000]
    promos_every_n_years:tuple=[8,4,2]
    cost_of_living_raise:tuple=[0.01,0.02,0.03]
    promo_raise: tuple=[0.07,0.15,0.2]
    savings_rate: tuple=[0.15,0.25,0.35]
    interest_rate: tuple=[0.03,0.05,0.06]
    desired_cash: int =1500000
    n_inter:int=1000
data=model_data=ModelInputs()


# In[31]:


normal_probability=1-data.recession_prob-data.expansion_prob
case_num=random.choices([0,1,2],weights=[data.recession_prob,normal_probability,data.expansion_prob])
case_num


# In[32]:


def get_economy_case_number(data):
    normal_probability=1-data.recession_prob-data.expansion_prob
    case_num=random.choices([0,1,2],weights=[data.recession_prob,normal_probability,data.expansion_prob])[0]
    return case_num
case=get_economy_case_number(data)
case


# In[34]:


def salary_at_year(data: ModelInputs, year,case):
    """
    Gets the salary at a given year from the start of the model based on cost of living raises and regular promotions.
    """
    # Every n years we have a promotion, so dividing the years and taking out the decimals gets the number of promotions
    num_promos = int(year / data.promos_every_n_years[case])
    
    # This is the formula above implemented in Python
    salary_t = data.starting_salary[case] * (1 + data.cost_of_living_raise[case]) ** year * (1 + data.promo_raise[case]) ** num_promos
    return salary_t
def cash_saved_during_year_case(data: ModelInputs, year,case):
    """
    Calculated the cash saved within a given year, by first calculating the salary at that year then applying the 
    savings rate.
    """
    salary = salary_at_year(data, year,case)
    cash_saved = salary * data.savings_rate[case]
    return cash_saved
def wealth_at_year_case(data: ModelInputs, year, prior_wealth,case):
    """
    Calculate the accumulated wealth for a given year, based on previous wealth, the investment rate,
    and cash saved during the year.
    """
    cash_saved = cash_saved_during_year_case(data, year,case)
    wealth = prior_wealth * (1 + data.interest_rate[case]) + cash_saved
    return wealth
def years_to_retirement_rf(data: ModelInputs,print_output=True):
    
    # starting with no cash saved
    prior_wealth = 0  
    wealth = 0
    
    year = 0  # will become 1 on first loop
    
    if print_output:
        print('Wealths over time:') # \n makes a blank line in the output.
    while wealth < data.desired_cash:
        year = year + 1
        case=get_economy_case_number(data)
        case_type=data.case_name[case]
        wealth = wealth_at_year_case(data, year, prior_wealth,case)
        if print_output:
            print(f'The wealth at year {year} is ${wealth:,.0f}.')

        # Set next year's prior wealth to this year's wealth
        prior_wealth = wealth
        
    # Now we have exited the while loop, so wealth must be >= desired_cash. Whatever last year was set
    # is the years to retirement.
    if print_output:
        print(f'\nRetirement:\nIt will take {year} years to retire.')  # \n makes a blank line in the output.
    return year


# In[35]:


years_to_retirement_rf(model_data)


# In[36]:


import pandas as pd


# In[37]:


def year_to_retirement_case_df(data):
    all_ytr=[]
    for i in range(data.n_inter):
        ytr=years_to_retirement_rf(data,print_output=False)
        all_ytr.append(ytr)
        df=pd.DataFrame()
        df['year to retirement']=all_ytr
    return df
df=year_to_retirement_case_df(data)
df


# In[38]:


def sumarize_ytr_df(df):
    avg_ytr=df['year to retirement'].mean()
    std_ytr=df['year to retirement'].std()
    max_ytr=df['year to retirement'].max()
    min_ytr=df['year to retirement'].min()
    print(f' it will take {avg_ytr:.0f} years to retire on average, with standard deviation{std_ytr:.1f}, max of{max_ytr:.0f} and min of {min_ytr:.0f}')
    


# In[39]:


def year_to_retirement_case_and_sumarize(data):
    df=year_to_retirement_case_df(data)
    sumarize_ytr_df(df)
year_to_retirement_case_and_sumarize(data)


# # Monte_Cartlo_Simulation

# In[56]:


@dataclass
class SimluationInputs:
    n_iterations: int = 10000
    starting_salary_std: int = 10000
    promos_every_n_years_std: int = 1.5
    cost_of_living_raise_std: float = 0.005
    promo_raise_std: float = 0.05
    savings_rate_std: float = 0.07
    interest_rate_std: float = 0.01
   
        
sim_data = SimluationInputs()


# In[57]:


@dataclass
class ModelInputs:
    starting_salary: int = 60000
    promos_every_n_years: int = 5
    cost_of_living_raise: float = 0.02
    promo_raise: float = 0.15
    savings_rate: float = 0.25
    interest_rate: float = 0.05
    years_to_retirement:int=24
    cash_spend_during_retirement:float=70000
        
model_data = ModelInputs()
data=model_data
data


# In[58]:


def random_normal_positive(mean, std):
    """
    This function keeps drawing random numbers from a normal distribution until it gets a positive number,
    then it returns that number.
    """
    drawn_value = -1  # initialize to some negative number so that the while loop will start
    while drawn_value < 0:
        drawn_value = random.normalvariate(mean, std)
    return drawn_value

def years_to_retirement_simulation_inputs(data, sim_data):
    """
    Randomly picks values from normal distributions for:
    - starting salary 
    - promotions every N years
    - cost of living raise
    - promotion raise
    - savings rate
    - interest rate
    
    These inputs are drawn based on using the ModelInputs values
    as means, and the _std values in SimulationInputs as standard deviations.
    
    Additionally, if any drawn value is zero or below, it will be drawn again.
    """
    starting_salary = random_normal_positive(data.starting_salary, sim_data.starting_salary_std)
    promos_every_n_years = random_normal_positive(data.promos_every_n_years, sim_data.promos_every_n_years_std)
    cost_of_living_raise = random_normal_positive(data.cost_of_living_raise, sim_data.cost_of_living_raise_std)
    promo_raise = random_normal_positive(data.promo_raise, sim_data.promo_raise_std)
    savings_rate = random_normal_positive(data.savings_rate, sim_data.savings_rate_std)
    interest_rate = random_normal_positive(data.interest_rate, sim_data.interest_rate_std)
    
    return (
        starting_salary,
        promos_every_n_years,
        cost_of_living_raise,
        promo_raise,
        savings_rate,
        interest_rate,
    )
    


# In[59]:


def cash_saved_during_year(data: ModelInputs, year):
    """
    Calculated the cash saved within a given year, by first calculating the salary at that year then applying the 
    savings rate.
    """
    salary = salary_at_year(data, year)
    cash_saved = salary * data.savings_rate
    return cash_saved
def wealth_at_year(data: ModelInputs, year, prior_wealth):
    """
    Calculate the accumulated wealth for a given year, based on previous wealth, the investment rate,
    and cash saved during the year.
    """
    cash_saved = cash_saved_during_year(data, year)
    wealth = prior_wealth * (1 + data.interest_rate) + cash_saved
    return wealth
def salary_at_year(data: ModelInputs, year):
    """
    Gets the salary at a given year from the start of the model based on cost of living raises and regular promotions.
    """
    # Every n years we have a promotion, so dividing the years and taking out the decimals gets the number of promotions
    num_promos = int(year / data.promos_every_n_years)
    
    # This is the formula above implemented in Python
    salary_t = data.starting_salary * (1 + data.cost_of_living_raise) ** year * (1 + data.promo_raise) ** num_promos
    return salary_t


# In[60]:


def years_to_retirement(data: ModelInputs, print_output=False):
    
    # starting with no cash saved
    prior_wealth = 0  
    wealth = 0
    
    year = 0  # will become 1 on first loop
    desired_cash=get_desired_cash(model_data)
    if print_output:
        print('Wealths over time:') # \n makes a blank line in the output.
    while wealth < desired_cash:
        year = year + 1
        wealth = wealth_at_year(data, year, prior_wealth)
        if print_output:
            print(f'The wealth at year {year} is ${wealth:,.0f}.')

        # Set next year's prior wealth to this year's wealth
        prior_wealth = wealth
        
    # Now we have exited the while loop, so wealth must be >= desired_cash. Whatever last year was set
    # is the years to retirement.
    if print_output:
        print(f'\nRetirement:\nIt will take {year} years to retire.')  # \n makes a blank line in the output.
    return year


# In[61]:


def years_to_retirement_single_simulation(data, sim_data):
    """
    Runs a single Monte Carlo simulation of the years to retirement model.
    
    Uses years_to_retirement_simulation_inputs
    """
    # Draw values of inputs from normal distributions
    (
        starting_salary,
        promos_every_n_years,
        cost_of_living_raise,
        promo_raise,
        savings_rate,
        interest_rate,
    ) = years_to_retirement_simulation_inputs(data, sim_data)
    
    # Construct model inputs
    new_data = ModelInputs(
        starting_salary=starting_salary,
        promos_every_n_years=promos_every_n_years,
        cost_of_living_raise=cost_of_living_raise,
        promo_raise=promo_raise,
        savings_rate=savings_rate,
        interest_rate=interest_rate,
    )

    # Run model
    ytr = years_to_retirement(new_data)
    
    return (
        starting_salary,
        promos_every_n_years,
        cost_of_living_raise,
        promo_raise,
        savings_rate,
        interest_rate,
        ytr
    )


# In[62]:


years_to_retirement_single_simulation(model_data, sim_data)


# ## Running the Full Monte Carlo Simulation

# In[63]:


def years_to_retirement_mc(data, sim_data):
    """
    Runs the full Monte Carlo simulation using years_to_retirement_single_simulation for 
    the n_iterations in the SimulationData.
    
    Outputs a DataFrame containing the inputs values as well as the years to retirement.
    """
    all_results = [years_to_retirement_single_simulation(data, sim_data) for i in range(sim_data.n_iterations)]
    df = pd.DataFrame(
        all_results,
        columns=[
            'Starting Salary', 
            'Promotions Every $N$ Years', 
            'Cost of Living Raise', 
            'Promotion Raise', 
            'Savings Rate', 
            'Interest Rate',
            'Years to Retirement'
        ]
    )
    return df


# In[64]:


df = years_to_retirement_mc(model_data, sim_data)


# In[65]:


def styled_df(df):
    """
    Styles DataFrames containing the inputs and years to retirement.
    """
    return df.style.format({
        'Starting Salary': '${:,.0f}', 
        'Promotions Every $N$ Years': '{:.1f}', 
        'Cost of Living Raise': '{:.2%}', 
        'Promotion Raise': '{:.2%}', 
        'Savings Rate': '{:.2%}', 
        'Interest Rate': '{:.2%}',
        'Years to Retirement': '{:.0f}'
    }).background_gradient(cmap='RdYlGn_r', subset='Years to Retirement')


# In[66]:


styled_df(df.head())


# ### Inputs and Years to Retirement Probability Table

# In[67]:


quants = df.quantile([i / 20 for i in range(1, 20)])
styled_df(quants)


# ## Analyze Results
# 
# ### Multivariate Regression
# 
# Now I will see how much each input affects each output, once all the inputs are varied in the simulation.

# In[68]:


input_cols = [
    'Starting Salary', 
    'Promotions Every $N$ Years', 
    'Cost of Living Raise', 
    'Promotion Raise', 
    'Savings Rate', 
    'Interest Rate',
]

for col in input_cols:
    df.plot.scatter(y='Years to Retirement', x=col)


# In[69]:


output_col = 'Years to Retirement'

X = sm.add_constant(df[input_cols])
y = df[output_col]

mod = sm.OLS(y, X)
result = mod.fit()
result.summary()


# We can see based on the p-values that all of the inputs are significantly related to the outputs. But we must incorporate the standard deviations of the inputs to understand which has the greatest impact.

# In[70]:


df.std()


# In[71]:


result.params* df.std()


# Savings rate seems to have the greatest impact. A one standard deviation (7%) increase in savings rate decreases years to retirement by 4.2 years. Cost of living raises are the least impactful with a one standard deviation increase (0.5%) in cost of living raises only decreasing years to retirement by 1.58 years.

# In[ ]:




