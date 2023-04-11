import streamlit as st
import streamlit_authenticator as stauth
import streamlit.components.v1 as components
import io
#import os
from dataclasses import dataclass
#import pickle
from pathlib import Path 
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import altair as alt
from googletrans import Translator
import pandas as pd
from PIL import Image
import numpy as np
import yaml
from yaml import load, dump
import time 
import random
import markdown
from typing import Optional, Tuple
#==================================================================================================================================


#==================================================================================================================================
np.random.seed(12345)
st.set_page_config(page_title="NTD Endgame", layout="wide")

with st.spinner("Loading NTD Endgame Tool ....."):
    time.sleep(1)

st.title("Neglected Tropical Diseases Costing Tool v.1.")

#================================================================================
# Functions to be used during the analyses

@st.cache

def mda_rounds_left(admin_level, level, status = "LF_MDA_left"):
    """
    Gets remaining MDA rounds that are then used as default values
    """
    unit_mda_status = max(list(set(country_espen[status][country_espen[admin_level] == level])))
    return unit_mda_status

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    # List of column types to check
    types = ["float16", "float32", "int8", "int16", "int32", "uint8", "uint16", "uint32"]
    
    # Iterate through all columns
    for col in df.columns:
        col_type = df[col].dtype
        
        # If column type is datetime, convert to datetime
        if col_type == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        
        # If column type is categorical, convert to categorical
        elif col_type == "object":
            df[col] = df[col].astype("category")
        
        # If column type is numerical, downcast to the smallest possible type
        elif col_type in types:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min >= 0:
                    if c_max <= 255:
                        df[col] = df[col].astype("uint8")
                    elif c_max <= 65535:
                        df[col] = df[col].astype("uint16")
                    elif c_max <= 4294967295:
                        df[col] = df[col].astype("uint32")
                    else:
                        df[col] = df[col].astype("uint64")
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype("float16")
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype("float32")
                else:
                    df[col] = df[col].astype("float64")
    
    return df


def download_chart(chart):
    """
    Adds a download button for a Matplotlib figure or an Altair chart to a Streamlit app.

    Parameters:
        chart (Union[plt.Figure, alt.Chart]): The chart to download.

    Returns:
        None
    """

    if isinstance(chart, plt.Figure):
        # save the figure to a bytes buffer
        buf = io.BytesIO()
        chart.savefig(buf, format='png', dpi=300, bbox_inches="tight")
        buf.seek(0)

        # create a download link
        st.download_button(
            label='Download Figure',
            data=buf,
            file_name='figure.png',
            mime='image/png'
        )

    elif isinstance(chart, alt.Chart):
        # save the chart to a JSON string
        chart_json = chart.to_json(indent=None)

        # create a bytes buffer with the chart data
        buf = io.BytesIO()
        buf.write(chart_json.encode('utf-8'))
        buf.seek(0)

        # create a download link
        st.download_button(
            label='Download Chart',
            data=buf,
            file_name='chart.json',
            mime='application/json'
        )

    else:
        raise TypeError("Invalid chart type. Must be a Matplotlib figure or an Altair chart.")

@st.cache
def cea_threshold(gdp_ppp_x=1, uk_cet = 26_705, gdp_ppp_uk =46_659, elasticity=2.478):
    """
    Function to estimate default country CEA thresholds
    We follow the approach in [Woods et al (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5193154/)
    The elasticity factor is based on [Milligan et al (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0001457514001687?via%3Dihub)
    These are more conservative than the ubiquitous and much-maligned 1-3 * GDP-based approaches
    The UK CEA threshold in 2013 according to [Claxton et al (2015)](https://pubmed.ncbi.nlm.nih.gov/25692211/) was 
    £12,936 pounds, which translates to £14,277 in 2020
    Spot conversion for £ to $ was 1.2837, and PPP adjustment was .688058
    ..........................................
    Attributes
    gdp_ppp_uk:                 int
                                    GDP PPP UK 2020
    elasticity:                 float
                                    Elasticity rate to convert thresholds across countries
    uk_cet:                     int
                                    Cost-effectiveness threshold UK 2020 based on calculations from Woods et al
    """
    gdp_ppp_x = country_inputs["Annual_PPP(Int$)"][country_inputs["Country"]==country].values[0]
    cet_threshold = uk_cet *((gdp_ppp_x/gdp_ppp_uk)**elasticity)

    return cet_threshold

def coverage_issues(country, lower=0, upper=60):
    """
    This function finds units with sub-optimal coverage
    These units are then used in the write statement to indicate problematic units
    .................
    Attributes
    --------------------------------------------------
    mda_coverage:           int
                                value from the MDA coverage slider
    """
    low_coverage = sorted(list(country_espen["IUs_NAME"][(country_espen["ADMIN0"]==country)
                        & (country_espen["Cov"] > lower) 
                        & (country_espen["Cov"] < upper)
                        &(country_espen["PopReq"] !=0)
                            ]))
    return low_coverage


def transmission_efficacy(country_status):
    """
    This function takes the MDA program status and calibrates transmission transmission_efficacy
    For example MDA 2 means that LF transmission drops to 25% of baseline rates
    NB: ensure formula works for future iterations
    ......................................
    Attributes
    -----------------------------------------
    mda_coverage:               str
                                    MDA coverage from drop down menu
    """   
    efficacy_dict = {"No program": 1,
                    "MDA 1": .5,
                    "MDA 2": .25,
                    "MDA 3": .12,
                    "MDA 4": .06,
                    "MDA 5": .05,
                    "MDA 6": .01}
    return efficacy_dict.get(country_status, 0)

@st.cache
def caseloads(at_risk_pop, time_horizon):
    """
    This function calculates the number of ADL cases p.a.
    Assumes all hydrocele and lymphedema cases are acquired at baseline (median age)
    It takes as an input the at-risk-population in a given year as the key input
    It assumes that an individual can either have hydrocele or lymphedema
    Assume 780 ADL per 1000 infected cases p.a.
    ................................
    Attribute
    at_risk_population:          int
                                    At risk population 
    -------------------------------------------------
    """
    potential_infections = at_risk_pop * 0.1
    clinical_disease = potential_infections * .33
    hydrocele_cases = clinical_disease * .625
    lymphedema_cases = clinical_disease * .375
    initiator = transmission_efficacy(country_status)
    mda_efficacy = [1, .5, .25, .12, .06, .05, .01] # Efficacy of MDA on reducing ADL
    rate_initiator = mda_efficacy.index(initiator)
    total_adl_cases = 0
    mda_0_adl = potential_infections * .78 * mda_efficacy[0]
    mda_1_adl = mda_0_adl * .78 * time_horizon * mda_efficacy[1]
    mda_2_adl = mda_1_adl * .78 * time_horizon * mda_efficacy[2]
    mda_3_adl = mda_2_adl * .78 * time_horizon * mda_efficacy[3]
    mda_4_adl = mda_3_adl * .78 * time_horizon * mda_efficacy[4]
    mda_5_adl = mda_4_adl * .78 * time_horizon * mda_efficacy[5]
    mda_6_adl = mda_5_adl * .78 * time_horizon * mda_efficacy[6]
    
    if rate_initiator == 0:
        total_adl_cases = mda_0_adl * time_horizon
    if rate_initiator == 1:
        total_adl_cases = mda_2_adl + mda_3_adl + mda_4_adl + mda_5_adl + mda_6_adl * (time_horizon - 4)
    if rate_initiator == 2:
        total_adl_cases = mda_3_adl + mda_4_adl + mda_5_adl + mda_6_adl * (time_horizon - 3)
    if rate_initiator == 3:
        total_adl_cases = mda_4_adl + mda_5_adl + mda_6_adl * (time_horizon - 2)
    if rate_initiator == 4:
        total_adl_cases = mda_5_adl + mda_6_adl * (time_horizon - 1)
    if rate_initiator == 5:
        total_adl_cases = mda_6_adl * (time_horizon - 1)
    
    y0 = [hydrocele_cases * .5]
    t = np.linspace(0,39, 40)
    r = 0.015
    K = [x * .05 for x in y0]
    params = [r, K]

    def hydrocele_decrement(y0, t, params):
        "Assumes logistic growth in hydrocele surgeries"
        X = y0[0]
        r = params[0]
        K = params[1]
        dXdt = r*X * (1 - X/K)
        return dXdt
    
    hydr_cum_growth = odeint(hydrocele_decrement, y0, t, args=(params,))
    hydr_cum_growth = hydr_cum_growth[:, 0]
    hydr_cum_decline =np.repeat(K, 40)
    hydr_remain_cases = hydr_cum_growth - hydr_cum_decline

    def basic_plot(t, hydr_remain_cases):
        t += 2020
        
        df = pd.DataFrame(list(zip(t, hydr_remain_cases)), columns=["Time", "Remaining Cases"])
        line_chart = alt.Chart(df).mark_line().encode(
            y = alt.Y("Remaining Cases"),
            x = alt.X("Time:N", title="Years")
        ).properties(
            title="Estimated decline in the demand for hydrocele cases with surgery"
        )
        return line_chart
    hydrocele_decline = basic_plot(t, hydr_remain_cases)
    return hydrocele_cases, lymphedema_cases, total_adl_cases, rate_initiator, hydrocele_decline

def daily_wage():
    """
    Function for daily value of productivity
    Adjust hourly wage by income inequality to get adjusted hourly wage
    """
    annual_gdp = country_inputs['Annual_PPP(Int$)'][country_inputs["Country"]==country]
    inequality_adjustment = country_inputs['inequality_.2_quintile'][country_inputs["Country"]==country]/100
    adj_daily_wage = annual_gdp * inequality_adjustment / (0.2 * 300) 
    adj_hourly_wage = adj_daily_wage * 5 / country_inputs["Weekly_Work_Hours"][country_inputs["Country"]==country]
    return adj_daily_wage, adj_hourly_wage


def random_normal_positive(float, std):
    """
    This function keeps drawing random numbers from a normal distribution until it gets a positive number,
    then it returns that number.
    """
    drawn_value = -1  # initialize to some negative number so that the while loop will start
    while drawn_value < 0:
        drawn_value = random.normalvariate(float, std)
    return drawn_value

def random_lognormal_positive(float, std):
    """
    This function keeps drawing random numbers from a lognormal distribution until it gets a positive number,
    then it returns that number.
    """
    drawn_value = -1  # initialize to some negative number so that the while loop will start
    while drawn_value < 0:
        drawn_value = np.random.lognormal(float, std)
    return drawn_value
def gamma_positive(float, std):
    """
    This function draws random numbers from a gamma distribution until it gets a positive number,
    then it returns that number
    """
    sample_mean = float
    sample_var = std ** 2
    shape = (sample_mean ** 2)/sample_var
    scale = sample_var/sample_mean
    drawn_value = -1
    while drawn_value < 0:
        drawn_value = np.random.gamma(shape, scale)
    return drawn_value

def gamma_simulator(mean: float, ci_lower: float, ci_upper: float, n_simulations: int = 1000):
    """
    Simulates n_simulations samples from a gamma distribution based on a given mean and confidence interval.

    Parameters
    ----------
    mean : float
        The mean of the gamma distribution.
    ci_lower : float
        The lower bound of the confidence interval for the mean.
    ci_upper : float
        The upper bound of the confidence interval for the mean.
    n_simulations : int, optional
        The number of gamma samples to simulate. Default is 10000.

    Returns
    -------
    np.ndarray
        An array of shape (n_simulations,) containing gamma samples with the specified mean and confidence interval.

    Notes
    -----
    The gamma distribution parameters are estimated from the mean and standard error of the mean of the original data.
    Specifically, the shape parameter is estimated as mean^2 / (std_error)^2, and the scale parameter is estimated as
    (std_error)^2 / mean, where std_error is half the width of the confidence interval (assuming a normal distribution).

    """
    std_error = (ci_upper - ci_lower)/(2 * 1.96)
    shape = mean**2/ std_error**2
    scale = std_error**2/mean
    gamma_samples = np.random.gamma(shape, scale, n_simulations)
    
    return gamma_samples

def age_distribution(float, std):
    """
    This function keeps drawing random numbers from a normal distribution until it gets a positive number,
    then it returns that number.
    """
    drawn_value = -1  # initialize to some negative number so that the while loop will start
    while drawn_value < 10 or drawn_value > 70:
        drawn_value = random.normalvariate(float, std)
    return drawn_value


def model_simulation_inputs(sim_data):
    """
    Pick random values from a a random normal distribution
    Constrain them to be positive
    """
    hosp_stay_surg = random_normal_positive(sim_data.hosp_stay_surg, sim_data.hosp_stay_surg_std)
    hydr_prop_pa = random_normal_positive(sim_data.hydr_prop_pa, sim_data.hydr_prop_pa_std)
    hydr_surgeries_perc = random_normal_positive(sim_data.hydr_surgeries_perc, sim_data.hydr_surgeries_perc_std)
    surgical_review_meetings = random_normal_positive(sim_data.surgical_review_meetings, sim_data.surgical_review_meetings_std)
    surgical_length_of_stay = random_normal_positive(sim_data.surgical_length_of_stay, sim_data.surgical_length_of_stay_std)
    perc_hydr_visits_non_adl = random_normal_positive(sim_data.perc_hydr_visits_non_adl, sim_data.perc_hydr_visits_non_adl_std)
    perc_adl_seek_rx = random_normal_positive(sim_data.perc_adl_seek_rx, sim_data.perc_adl_seek_rx_std)
    adl_episodes_pa_no_mda = random_normal_positive(sim_data.adl_episodes_pa_no_mda, sim_data.adl_episodes_pa_no_mda_std)
    perc_lymph_adl_pa = random_normal_positive(sim_data.perc_lymph_adl_pa, sim_data.perc_lymph_adl_pa_std)
    lymphedema_cases = random_normal_positive(sim_data.lymphedema_cases, sim_data.lymphedema_cases_std)
    perc_hydr_adl_pa = random_normal_positive(sim_data.perc_hydr_adl_pa, sim_data.perc_hydr_adl_pa_std)
    hydrocele_cases = random_normal_positive(sim_data.hydrocele_cases, sim_data.hydrocele_cases_std)
    
    operation_time = random_normal_positive(sim_data.operation_time, sim_data.operation_time_std)
    prop_dressed = random_normal_positive(sim_data.prop_dressed, sim_data.prop_dressed_std)
    clinic_consultation_time = sim_data.clinic_consultation_time
    
    perc_lymph_visits_non_adl = random_normal_positive(sim_data.perc_lymph_visits_non_adl, sim_data.perc_lymph_visits_non_adl_std)
    daly_weight_hydr = random_normal_positive(sim_data.daly_weight_hydr, sim_data.daly_weight_hydr_std)
    morbidity_reduction_surg = gamma_positive(sim_data.morbidity_reduction_surg, sim_data.morbidity_reduction_surg_std)
    surg_success_rate = gamma_positive(sim_data.surg_success_rate, sim_data.surg_success_rate_std)
    age_hydr_sur = age_distribution(sim_data.age_hydr_sur, sim_data.age_hydr_sur)
    r = random_normal_positive(sim_data.r, sim_data.r_std)
    hydr_reduction_mda = gamma_positive(sim_data.hydr_reduction_mda, sim_data.hydr_reduction_mda_std)
    lymph_reduction_mda = gamma_positive(sim_data.lymph_reduction_mda, sim_data.lymph_reduction_mda_std)
    
    prod_loss_chronic_hydro = random_normal_positive(sim_data.prod_loss_chronic_hydro, sim_data.prod_loss_chronic_hydro_std)
    disability_weight_hydro = random_normal_positive(sim_data.disability_weight_hydro, sim_data.disability_weight_hydro_std)
    disability_weight_adl = random_normal_positive(sim_data.disability_weight_adl, sim_data.disability_weight_adl_std)
    disability_weight_lymph = random_normal_positive(sim_data.disability_weight_lymph, sim_data.disability_weight_lymph_std)
    prod_loss_chronic_lymph = random_normal_positive(sim_data.prod_loss_chronic_lymph, sim_data.prod_loss_chronic_lymph_std)
    prod_loss_adl_episode = random_normal_positive(sim_data.prod_loss_adl_episode, sim_data.prod_loss_adl_episode_std)
    days_adl_episode = random_normal_positive(sim_data.days_adl_episode, sim_data.days_adl_episode_std)
    
    adl_reduction_mda = random_normal_positive(sim_data.adl_reduction_mda, sim_data.adl_reduction_mda_std)
    return (
        hydrocele_cases, lymphedema_cases, perc_hydr_adl_pa, adl_episodes_pa_no_mda, perc_adl_seek_rx,
        perc_hydr_visits_non_adl, surgical_length_of_stay, surgical_review_meetings,
        hydr_surgeries_perc, hydr_prop_pa, hosp_stay_surg, operation_time, prop_dressed,
        clinic_consultation_time, daly_weight_hydr, morbidity_reduction_surg,surg_success_rate,
        age_hydr_sur, r, hydr_reduction_mda, prod_loss_chronic_hydro, disability_weight_hydro,
        lymph_reduction_mda, adl_reduction_mda, prod_loss_chronic_lymph, prod_loss_adl_episode,
        perc_lymph_visits_non_adl, disability_weight_lymph, disability_weight_adl,
        days_adl_episode, perc_lymph_adl_pa
    )

def hydrocele_decrement(y0, t, params):
    """
    Assumes logistic growth in hydrocele surgeries
    """
    X = y0[0]
    r = params[0]
    K = params[1]
    dXdt = r*X * (1 - X/K)
    return dXdt

def model_single_run(sim_data):
    (
    hydrocele_cases, lymphedema_cases, perc_hydr_adl_pa, adl_episodes_pa_no_mda, perc_adl_seek_rx,
    perc_hydr_visits_non_adl, surgical_length_of_stay, surgical_review_meetings,
    hydr_surgeries_perc, hydr_prop_pa, hosp_stay_surg, operation_time, prop_dressed,
    clinic_consultation_time, daly_weight_hydr, morbidity_reduction_surg,surg_success_rate,
    age_hydr_sur, r, hydr_reduction_mda, prod_loss_chronic_hydro, disability_weight_hydro,
    lymph_reduction_mda, adl_reduction_mda, prod_loss_chronic_lymph, prod_loss_adl_episode,
    perc_lymph_visits_non_adl, disability_weight_lymph, disability_weight_adl,
    days_adl_episode, perc_lymph_adl_pa
    ) = model_simulation_inputs(sim_data)
    
    annual_ip_days = hydrocele_cases * hydr_surgeries_perc * hydr_prop_pa * hosp_stay_surg
    
    annual_review_days = (hydrocele_cases * hydr_surgeries_perc *  
                        hydr_prop_pa * clinic_consultation_time * 
                        surgical_review_meetings)
    
    theatre_time = (hydrocele_cases * hydr_surgeries_perc * hydr_prop_pa * operation_time) # Assumes 1 hour per procedure including prep-time
    
    surgeon_time = theatre_time + annual_review_days
    
    annual_adl_days = (hydrocele_cases * 
                    perc_hydr_adl_pa * 
                    adl_episodes_pa_no_mda * 
                    perc_adl_seek_rx *
                    clinic_consultation_time)
    
    annual_non_adl_days = (hydrocele_cases *
                        perc_hydr_visits_non_adl *
                        clinic_consultation_time)
    
    community_hydr_care = (hydrocele_cases * prop_dressed * 12 * clinic_consultation_time)
    
    annual_health_sector_time = (annual_ip_days + 
                                surgeon_time + 
                                annual_adl_days + 
                                annual_non_adl_days + 
                                community_hydr_care)
    
    return     (
        hydrocele_cases, 
        lymphedema_cases,
        perc_hydr_adl_pa, adl_episodes_pa_no_mda, perc_adl_seek_rx,
        perc_hydr_visits_non_adl, surgical_length_of_stay, surgical_review_meetings,
        hydr_surgeries_perc, hydr_prop_pa, hosp_stay_surg, operation_time, prop_dressed,
        clinic_consultation_time, daly_weight_hydr, morbidity_reduction_surg,surg_success_rate,
        age_hydr_sur, r, hydr_reduction_mda, prod_loss_chronic_hydro, disability_weight_hydro,
        lymph_reduction_mda, adl_reduction_mda, prod_loss_chronic_lymph, prod_loss_adl_episode,
        perc_lymph_visits_non_adl, disability_weight_lymph, disability_weight_adl,
        days_adl_episode, perc_lymph_adl_pa, annual_ip_days, annual_review_days, surgeon_time, theatre_time,
        annual_adl_days, annual_non_adl_days, community_hydr_care, annual_health_sector_time,  
    )


def monte_carlo_data(sim_data):
    values = [model_single_run(sim_data) for i in range(sim_data.n_iterations)]
    df = pd.DataFrame(
        values,
        columns = [ 
            "hydrocele_cases", "lymphedema_cases", "perc_hydr_adl_pa", "adl_episodes_pa_no_mda", "perc_adl_seek_rx",
            "perc_hydr_visits_non_adl", "surgical_length_of_stay", "surgical_review_meetings",
            "hydr_surgeries_perc", "hydr_prop_pa", "hosp_stay_surg", "operation_time", "prop_dressed",
            "clinic_consultation_time", "daly_weight_hydr", "morbidity_reduction_surg", "surg_success_rate",
            "age_hydr_sur", "r", "hydr_reduction_mda", "prod_loss_chronic_hydro", "disability_weight_hydro",
            "lymph_reduction_mda", "adl_reduction_mda", "prod_loss_chronic_lymph", "prod_loss_adl_episode",
            "perc_lymph_visits_non_adl", "disability_weight_lymph", "disability_weight_adl",
            "days_adl_episode", "perc_lymph_adl_pa", "annual_ip_days", "annual_review_days", "surgeon_time",
            "theatre_time", "annual_adl_days", "annual_non_adl_days", "community_hydr_care",
            "annual_health_sector_time"
        ]
    )
    return df

@st.cache
def hydr_single_run(sim_data):
    """
    Estimate remaining hydrocele cases with surgery per year
    Uses odeint package from scipy.integrate
    """
    hydrocele_cases = model_simulation_inputs(sim_data)[0]
    hydr_surgeries_perc = model_simulation_inputs(sim_data)[8]
    hydr_prop_pa = model_simulation_inputs(sim_data)[9]
    r = model_simulation_inputs(sim_data)[18]
    y0 = [hydrocele_cases * hydr_surgeries_perc]
    t = np.linspace(0,39, 40)
    K = [x *  hydr_prop_pa for x in y0]
    params = [r, K]
    hydr_cum_growth = odeint(hydrocele_decrement, y0, t, args=(params,))
    hydr_cum_growth = hydr_cum_growth[:, 0]
    hydr_cum_decline =np.repeat(K, 40)
    hydr_remain_cases = hydr_cum_growth - hydr_cum_decline
    return hydr_remain_cases


def hydrocele_plotter(sim_data):
    """
    Plots the remaining hydrocele cases per location
    """
    values = [hydr_single_run(sim_data) for i in range(10)]
    time = np.linspace(0,39,40) + 2020
    df = pd.DataFrame(values).T
    df = df.add_prefix("simulation_")
    df.insert(loc=0, column="year", value=time)
    variables = df.columns[1:]

    fig, ax = plt.subplots(figsize=(8,5), dpi=100)
    plt.grid(axis="both", alpha=.8, lw=.5)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for x in variables:
        plt.plot(df["year"], df[x], lw=.5)
        plt.axhline(0, ls="--", lw=.5)

    current_values = plt.gca().get_yticks()
    ax.set_yticks([x for x in current_values])
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])  
    plt.ylabel(translate_text("Estimated hydrocele surgical demand"))
    plt.title(translate_text("Simulated hydrocele surgical demand over time"))
    return fig

def styler(df):
    """
    Small function for styling pandas dataframe
    """
    return df.style.format({
        "Mean": "{:,.0f}",
        "Upper": "{:,.0f}",
        "Lower": "{:,.0f}"
    }).set_properties(**{'text-align': 'left'})

def show_df(dataset):
    """
    Show interactive datasets for country inputs
    """
    grids = GridOptionsBuilder.from_dataframe(dataset)
    grids.configure_pagination()
    grids.configure_selection(selection_mode="multiple", use_checkbox=True)
    grids.configure_side_bar()
    grids.configure_default_column(groupable=True, value=True, enableRowGroup=True,
                                aggFunc="sum", editable=True)
    gridOptions = grids.build()
    AgGrid(dataset, gridOptions=gridOptions, enable_enterprise_modules=True)
    
@st.cache
def annual_workdays_lost_no_mda(simulated_df,lymphedema_cases, hydrocele_cases):
    """
    Sum of workdays lost from chronic disease and from ADL episodes
    It takes in the dataset with Monte Carlo simulations
    """
    adl_hydrocele_no_mda_pa = (simulated_df["hydrocele_cases"] * 
                        simulated_df["perc_hydr_adl_pa"] * 
                        simulated_df["adl_episodes_pa_no_mda"] * 
                        simulated_df["days_adl_episode"] )
    adl_lymphedema_no_mda_pa = (simulated_df["lymphedema_cases"] * 
                            simulated_df["perc_lymph_adl_pa"] * 
                            simulated_df["adl_episodes_pa_no_mda"] *
                            simulated_df["days_adl_episode"] )
    adl_days_lost_no_mda_pa = adl_hydrocele_no_mda_pa + adl_lymphedema_no_mda_pa
    chronic_lymph_days_pa = simulated_df['prod_loss_chronic_lymph'] * 300 * lymphedema_cases
    chronic_hydro_days_pa = simulated_df['prod_loss_chronic_hydro'] * 300 * hydrocele_cases
    
    total_lower = (adl_hydrocele_no_mda_pa.quantile(.05)+
                adl_lymphedema_no_mda_pa.quantile(.05)+
                chronic_lymph_days_pa.quantile(.05)+
                chronic_hydro_days_pa.quantile(.05)
                )
    total_upper = (adl_hydrocele_no_mda_pa.quantile(.95)+
                adl_lymphedema_no_mda_pa.quantile(.95)+
                chronic_lymph_days_pa.quantile(.95)+
                chronic_hydro_days_pa.quantile(.95)
                )
    total_mean = (adl_hydrocele_no_mda_pa.mean()+
                adl_lymphedema_no_mda_pa.mean()+
                chronic_lymph_days_pa.mean()+
                chronic_hydro_days_pa.mean()
                )

    df = pd.DataFrame(columns = ["Category", "Mean", "Lower", "Upper"])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Hydrocele ADL days", 
                    "Mean": round(adl_hydrocele_no_mda_pa.mean()), 
                    "Lower":adl_hydrocele_no_mda_pa.quantile(.05), 
                    "Upper":adl_hydrocele_no_mda_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Lymphedema ADL days", 
                    "Mean": adl_lymphedema_no_mda_pa.mean(), 
                    "Lower":adl_lymphedema_no_mda_pa.quantile(.05), 
                    "Upper":adl_lymphedema_no_mda_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Chronic Hydrocele days", 
                    "Mean": chronic_hydro_days_pa.mean(), 
                    "Lower":chronic_hydro_days_pa.quantile(.05), 
                    "Upper":chronic_hydro_days_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Chronic Lymphedema days", 
                    "Mean": chronic_lymph_days_pa.mean(), 
                    "Lower":chronic_lymph_days_pa.quantile(.05), 
                    "Upper":chronic_lymph_days_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Total", 
                    "Mean": total_mean, 
                    "Lower":total_lower, 
                    "Upper":total_upper}])])

    df = df.reset_index(drop=True)
    
    economic_loss_mean = int(total_mean * daily_wage()[0])
    economic_loss_upper = int(total_upper * daily_wage()[0])
    economic_loss_lower = int(total_lower * daily_wage()[0])

    days_narrative = f"""
    In a non-MDA scenario, there would be {total_mean:,.0f} [{total_lower:,.0f}, {total_upper:,.0f}] productive days lost per year
    due to lymphatic filariasis at this administrative level annually. This includes workdays lost from adenolymphangitis as well as 
    productivity losses from chronic disease. This translates to approximately ${economic_loss_mean:,.0f} [{economic_loss_lower:,.0f}, 
    {economic_loss_upper:,.0f}] economic loss p.a.
    These values do not include other economic losses. See main narrative for this.

    """
    return df, days_narrative, adl_days_lost_no_mda_pa

@st.cache
def dalys_pa(hydrocele_cases, lymphedema_cases, simulated_df):
    """
    Estimate DALYs lost p.a. from cases
    """
    total_adl = annual_workdays_lost_no_mda(simulated_df,lymphedema_cases, hydrocele_cases)[2]
    hydrocele_daly = simulated_df['disability_weight_hydro'] * hydrocele_cases
    lymphedema_daly = simulated_df['disability_weight_lymph'] * lymphedema_cases
    adl_daly = simulated_df['disability_weight_adl'] * total_adl * simulated_df["days_adl_episode"]/300
    total_daly = hydrocele_daly + lymphedema_daly + adl_daly
    df = pd.DataFrame(columns = ["Category", "Mean", "Lower", "Upper"])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Hydrocele DALYs p.a.", 
                    "Mean": hydrocele_daly.mean(), 
                    "Lower":hydrocele_daly.quantile(.05), 
                    "Upper":hydrocele_daly.quantile(.95)}])])
    
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Lymphedema DALYs p.a.", 
                    "Mean": lymphedema_daly.mean(), 
                    "Lower":lymphedema_daly.quantile(.05), 
                    "Upper":lymphedema_daly.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "ADL DALYs p.a.", 
                    "Mean": adl_daly.mean(), 
                    "Lower":adl_daly.quantile(.05), 
                    "Upper":adl_daly.quantile(.95)}])])

    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Total DALYs p.a.", 
                    "Mean": total_daly.mean(), 
                    "Lower":total_daly.quantile(.05), 
                    "Upper":total_daly.quantile(.95)}])])
    df = df.reset_index(drop=True)
    return df

@st.cache
def annual_workdays_lost_with_mda(simulated_df,lymphedema_cases, hydrocele_cases):
    """
    Sum of workdays lost from chronic disease and from ADL episodes
    This does not incorporate community wound care which has a bigger impact than MDA
    It takes in the dataset with Monte Carlo simulations
    """
    
    adl_hydrocele_with_mda_pa = (simulated_df["hydrocele_cases"] * simulated_df['hydr_reduction_mda'] *
                                simulated_df['adl_reduction_mda'] * simulated_df["perc_hydr_adl_pa"] * 
                                simulated_df["adl_episodes_pa_no_mda"] * simulated_df["days_adl_episode"])
    
    adl_lymphedema_with_mda_pa = (simulated_df["lymphedema_cases"] * simulated_df['lymph_reduction_mda'] *
                            simulated_df['adl_reduction_mda'] * simulated_df["perc_lymph_adl_pa"] * 
                            simulated_df["adl_episodes_pa_no_mda"] * simulated_df["days_adl_episode"] )
    
    adl_days_lost_with_mda_pa = adl_hydrocele_with_mda_pa + adl_lymphedema_with_mda_pa

    chronic_lymph_days_pa = (simulated_df['prod_loss_chronic_lymph'] * 300 *
                            simulated_df['lymph_reduction_mda'] * lymphedema_cases)
    chronic_hydro_days_pa = (simulated_df['prod_loss_chronic_hydro'] * 300 *
                            simulated_df['hydr_reduction_mda'] * hydrocele_cases)
    
    total_lower = (adl_hydrocele_with_mda_pa.quantile(.05)+
                adl_lymphedema_with_mda_pa.quantile(.05)+
                chronic_lymph_days_pa.quantile(.05)+
                chronic_hydro_days_pa.quantile(.05)
                )
    total_upper = (adl_hydrocele_with_mda_pa.quantile(.95)+
                adl_lymphedema_with_mda_pa.quantile(.95)+
                chronic_lymph_days_pa.quantile(.95)+
                chronic_hydro_days_pa.quantile(.95)
                )
    total_mean = (adl_hydrocele_with_mda_pa.mean()+
                adl_lymphedema_with_mda_pa.mean()+
                chronic_lymph_days_pa.mean()+
                chronic_hydro_days_pa.mean()
                )
    
    df = pd.DataFrame(columns = ["Category", "Mean", "Lower", "Upper"])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Hydrocele ADL days", 
                    "Mean": round(adl_hydrocele_with_mda_pa.mean()), 
                    "Lower":adl_hydrocele_with_mda_pa.quantile(.05), 
                    "Upper":adl_hydrocele_with_mda_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Lymphedema ADL days", 
                    "Mean": adl_lymphedema_with_mda_pa.mean(), 
                    "Lower":adl_lymphedema_with_mda_pa.quantile(.05), 
                    "Upper":adl_lymphedema_with_mda_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Chronic Hydrocele days", 
                    "Mean": chronic_hydro_days_pa.mean(), 
                    "Lower":chronic_hydro_days_pa.quantile(.05), 
                    "Upper":chronic_hydro_days_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Chronic Lymphedema days", 
                    "Mean": chronic_lymph_days_pa.mean(), 
                    "Lower":chronic_lymph_days_pa.quantile(.05), 
                    "Upper":chronic_lymph_days_pa.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Total", 
                    "Mean": total_mean, 
                    "Lower":total_lower, 
                    "Upper":total_upper}])])

    df = df.reset_index(drop=True)
    #df = df.style.set_properties(**{'text-align': 'left'})
    
    days_narrative = f"""
    In a MDA scenario, there would be {total_mean:,.0f} [{total_lower:,.0f}, {total_upper:,.0f}] productive days lost per year
    due to lymphatic filariasis at this administrative level annually. This translates to approximately ${total_mean:,.2f} economic losses p.a.
    """
    return df, days_narrative, adl_days_lost_with_mda_pa

@st.cache
def dalys_pa_mda(hydrocele_cases, lymphedema_cases, simulated_df):
    """
    Estimate DALYs lost p.a. from cases
    """
    total_adl = annual_workdays_lost_with_mda(simulated_df,lymphedema_cases, hydrocele_cases)[2]
    hydrocele_daly = simulated_df['disability_weight_hydro'] * hydrocele_cases * simulated_df['hydr_reduction_mda']
    lymphedema_daly = simulated_df['disability_weight_lymph'] * lymphedema_cases * simulated_df['lymph_reduction_mda']
    adl_daly = simulated_df['disability_weight_adl'] * total_adl * simulated_df["days_adl_episode"]/300
    total_daly = hydrocele_daly + lymphedema_daly + adl_daly
    df = pd.DataFrame(columns = ["Category", "Mean", "Lower", "Upper"])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Hydrocele DALYs p.a.", 
                    "Mean": hydrocele_daly.mean(), 
                    "Lower":hydrocele_daly.quantile(.05), 
                    "Upper":hydrocele_daly.quantile(.95)}])])
    
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Lymphedema DALYs p.a.", 
                    "Mean": lymphedema_daly.mean(), 
                    "Lower":lymphedema_daly.quantile(.05), 
                    "Upper":lymphedema_daly.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "ADL DALYs p.a.", 
                    "Mean": adl_daly.mean(), 
                    "Lower":adl_daly.quantile(.05), 
                    "Upper":adl_daly.quantile(.95)}])])
    df = pd.concat([df, pd.DataFrame.from_records([{"Category": "Total DALYs p.a.", 
                    "Mean": total_daly.mean(), 
                    "Lower":total_daly.quantile(.05), 
                    "Upper":total_daly.quantile(.95)}])])
    df = df.reset_index(drop=True)
    return df

def patient_hours(time_horizon):
    """
    This function calculates the total patient hours from a health system perspective
    This is different from the total time the patient losses accessing care, waiting and also from 
    absenteeism and presenteeism.
    ...................................
    Attributes:
    hydrocele pop:                   float
                                        The number of patients with hydroceles weighted for severity
    non-adl hydrocele visits p.a.    float
                                        The number of non-ADL visits per year for hydrocele patients (include post-op reviews here)
    hydrocele surgeries p.a.         float
                                        Proportion of hydrocele patients undergoing surgery p.a.
    hydrocele surgeries lifetime:    float
                                        Proportion of hydrocele surgeries that undergo surgeries in a lifetime (calibrated by horizon)
    surgical length of stay:         float
                                        Duration of surgical stay in hospital (including pre and post-surgical)
    lymphedema population:             float
                                            Number of LF patients with lymphedema
    non-adl lymphedema visits p.a.   float
                                        Number of clinic visits e.g. for wound care per year. Lymphedema assumed to be life long if Stage 3+
    total adl population:            float
                                        Number of ADL visits p.a. (Assumed to be 780:1000 infected)
    time horizon:                     int
                                        Time horizon for analysis
    MDA phase:                        int
                                        MDA phase to reduce number of ADL cases. That is, after a certain period, no ADL visits will be expected.
    ----------------------------------------------------------------------------------------------
    """
    hydrocele_pop = caseloads(at_risk_pop, time_horizon)[0]
    lymphedema_pop = caseloads(at_risk_pop, time_horizon)[1]
    mda_phase = caseloads(at_risk_pop, time_horizon)[3]

    pass

@st.cache
def econ_values():
    """
    Returns the economic values for a given country, as entered by the user in the Streamlit interface.

    Returns:
    min_hwage (float): The minimum hourly wage for the given country, in Int$ adjusted for purchasing power parity.
    weekly_hours (float): The average number of weekly work hours for the given country.
    annual_ppp (float): The per capita income for the given country, in Int$ adjusted for purchasing power parity.
    inflation_rate (float): The inflation rate for the given country, expressed as a percentage.
    wtp_threshold (float): The willingness-to-pay threshold for the given country, as estimated by the World Bank.
    """
    min_hwage = st.number_input(f"Hourly wage (Int$): {country_dict[country]['Year']}",
                value=country_dict[country]["Hourly_PPP(Int$)"])
    weekly_hours = st.number_input("Weekly work hours",
                value=country_dict[country]['Weekly_Work_Hours'])
    annual_ppp = st.number_input("Per capita income PPP (Int$)",
                value=country_dict[country]['Annual_PPP(Int$)'])
    inflation_rate = st.number_input(f"Inflation rate: {country_dict[country]['Date of information']}",
                            value=country_dict[country]["Inflation rate (consumer prices) (%)"])
    wtp_threshold = country_dict[country]['wb_estimate']
    return min_hwage, weekly_hours, annual_ppp, inflation_rate, wtp_threshold

def cost_discounter(annual_costs, time_horizon, disc_costs = .04):
    """
    Function to discount costs and give cumulative costs
    ..............................
    Attributes
    -----------------------------------------------
    annual_costs:           float
                                list of annual program costs
    disc_costs:             float
                                discount rate for costs
    time_horizon:           int
                                time horizon for analysis from slider
    """
    cost_discounts = [(1 + disc_costs) ** x for x in range(1, time_horizon + 1)]
    yearly_costs = [val/disc for val,disc in zip(annual_costs, cost_discounts)]
    return np.sum(yearly_costs)

@st.cache
def unit_mda_coster(country, mda_cost, disc_costs):
    """
    This function returns the summarized MDA costs for a country
    It collates this by implementing unit and uses IU-CODES since they are unique
    The average cost of mdas per IU is done by country teams
    This enters the cost as an annual cost
    The function then uses the number of MDA rounds left for discounting
    ........................................
    Attributes
    country:             str
                            This is the country selected from the tool
    mda_cost:            float
                            IU MDA costs for one round
    disc_costs:          float
                            Discount rates for costs - default is 0.03
    -----------------------------------------------------
    """
    iu_code = list(set(country_espen["IU_CODE"][(country_espen["ADMIN0"]==country) & (country_espen["Endemicity"]!="Non-endemic")]))
    combo_mda_costs = []
    
    for x in iu_code:
        rounds_left = country_espen["LF_MDA_left"][country_espen["IU_CODE"]==x].values[0]
        if rounds_left == 0:
            unit_mda_costs = 0
            combo_mda_costs.append(unit_mda_costs) 
        elif rounds_left !=0:
            unit_mda_costs = cost_discounter(np.repeat(mda_cost, rounds_left), disc_costs, rounds_left)     
            combo_mda_costs.append(unit_mda_costs)    
    return sum(combo_mda_costs)


def confidence_ellipse(x, y, n_std=3.0, facecolor="none", **kwargs):
    """
    See full explanation here [Carsten Schelp, (2020)](https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html)
    """
    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    if x.size != y.size:
        raise ValueError("x and y must be of the same size")
    cov = np.cov(x,y)
    pearson = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0,0),
        width = ell_radius_x * 2,
        height = ell_radius_y * 2,
        facecolor = facecolor, **kwargs
    )
    scale_x = np.sqrt(cov[0,0] * n_std)
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1,1] * n_std)
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

@st.cache
def icers(costs: float, effects: float) -> float:
    """
    Function to model ICERs and generate an ICER plane
    Takes change in costs and change in effects
    Draws a CEA plane and CEAC curve
    ....................
    Attributes
    costs:      float (list)
                    1000 costs simulations
    effects:    float (list)
                    1000 effect simulations e.g., DALYs
    wtp_threshold:  float
                    Willingness to pay threshold -- calc. from UK and Country- GDP
    """
    cost_effects = [float(cost)/float(effect) if effect != 0 else np.nan for cost, effect in zip(costs, effects)]
    probs = []
    for i in range(len(cost_effects)):
        k = len([x for x in cost_effects if x < i])/len(cost_effects)
        probs.append(k)
    source = pd.DataFrame({
        "x": effects,
        "y": costs,
        "cea": cost_effects,
        "probs": probs,
        "wtp": list(range(len(costs)))
    })
    return source

def ceac(source, **kwargs):
    alt.Chart(source.reset_index()).mark_line().encode(
        x="wtp",
        y="probs"
    ).interactive()

def ceplane(source, wtp_threshold):
    point1 = [0, 0]
    point2 = [1, wtp_threshold]
    c_max = np.max(source["y"]) * 1.2/wtp_threshold
    x_values = [point1[0], point2[0] * c_max]
    y_values = [point1[1], point2[1] * c_max]
    probability = len([x for x in source["cea"] if x < wtp_threshold])/len(source["cea"])
    txt = f"There is {probability} that the intervention is cost effective a threshold of ${wtp_threshold} per QALY"
    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    plt.plot(x_values, y_values, label="WTP Threshold")
    plt.scatter(source["x"], source["y"], s=2, alpha=.5, label="ICER")
    confidence_ellipse(source["x"], source["y"], ax, edgecolor="red", label="95% CI")
    ax.axvline(0, linecolor="k")
    ax.axhline(0,linecolor="k")
    ax.xaxis.set_label_coords(1.0, 0.0)
    ax.yaxis.set_label_coords(0.0, 0.0)
    ax.spines["left"].set_position("data", 0)
    ax.spines["bottom"].set_position("data", 0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc="best")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10);
    plt.show()

def pop_distribution():
    """
    Population distribution 
    """
    avg_LE = st.number_input("Average life expectancy",
                        value=country_dict[country]['Life_Expectancy'])
    median_age_2020 = country_inputs["median_age_2020"][country_inputs["Country"] == country]
    mda_target = st.number_input("MDA Target Population",
                        value=pop_tag_mda)
    mda_target_pop = mda_target               
    _0_4 = st.number_input("Population 0 - 4 years",
                        value=mda_target_pop * 0)
    _5_9 = st.number_input("Population 5 - 9 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[1]/100))
    _10_14 = st.number_input("Population 10 - 14 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[2]/100))
    _15_19 = st.number_input("Population 15 - 19 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[3]/100))
    _20_24 = st.number_input("Population 20 - 24 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[4]/100))
    _25_29 = st.number_input("Population 25 - 29 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[5]/100))
    _30_34 = st.number_input("Population 30 - 34 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[6]/100))
    _35_39 = st.number_input("Population 35 - 39 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[7]/100))
    _40_44 = st.number_input("Population 40 - 44 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[8]/100))
    _45_49 = st.number_input("Population 45 - 49 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[9]/100))
    _50_54 = st.number_input("Population 50 - 54 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[10]/100))
    _55_59 = st.number_input("Population 55 - 59 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[11]/100))
    _60_64 = st.number_input("Population 60 - 64 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[12]/100))
    _65_69 = st.number_input("Population 65 - 69 years",
                        value=int(mda_target_pop * ssa_pop_pyramid[13]/100))
    _70_p = st.number_input("Population over 70",
                        value=int(mda_target_pop * ssa_pop_pyramid[14]/100))

@st.cache
def hydrocele_clinic_time_no_mda(simulated_df):
    """
    Estimate time spent by HCWs taking care of hydrocele patients
    """
    annual_ip_days = simulated_df["annual_ip_days"].mean()
    surgeon_time = simulated_df["surgeon_time"].mean()
    annual_adl_days = simulated_df["annual_adl_days"].mean()
    annual_non_adl_days = simulated_df["annual_non_adl_days"].mean()
    annual_health_sector_time = simulated_df["annual_health_sector_time"].mean()
    
    return annual_ip_days, surgeon_time, annual_adl_days, annual_non_adl_days, annual_health_sector_time

@st.cache
def hydrocele_clinic_time_mda(simulated_df):
    """
    Estimate time spent by HCWs taking care of hydrocele patients
    """
    annual_ip_days = (simulated_df["annual_ip_days"] * simulated_df['hydr_reduction_mda']).mean()
    surgeon_time = (simulated_df["surgeon_time"] * simulated_df['hydr_reduction_mda']).mean()
    annual_adl_days = (simulated_df["annual_adl_days"]* simulated_df['hydr_reduction_mda']).mean()
    annual_non_adl_days = (simulated_df["annual_non_adl_days"]*simulated_df['hydr_reduction_mda']).mean()
    annual_health_sector_time = (simulated_df["annual_health_sector_time"]*
                                    simulated_df['hydr_reduction_mda']*
                                simulated_df['adl_reduction_mda']).mean()
    return annual_ip_days, surgeon_time, annual_adl_days, annual_non_adl_days, annual_health_sector_time

@st.cache
def direct_health_sector_costs():
    """
    Estimate direct health system costs minus NTD program costs
    """
    medical_inflation = .04
    base_inflation = .09
    total_inflation = medical_inflation + base_inflation + 1
    total_surgeries = simulated_df["hydrocele_cases"].mean() * .5 * .05
    
    opd_days = (simulated_df["annual_review_days"] + 
                simulated_df["annual_adl_days"] + 
                simulated_df["annual_non_adl_days"])
    opd_costs = country_inputs["Primary Hospital_OPD_Costs"][country_inputs["Country"]== country] * (total_inflation) ** 10
    total_opd_costs = opd_costs * opd_days.mean()
    ipd_costs = country_inputs["Primary Hospital_IPD_Costs"][country_inputs["Country"]== country] * (total_inflation) ** 10
    total_ipd_costs =  (ipd_costs * simulated_df["annual_ip_days"]).mean()
    theatre_costs = ipd_costs * 5
    total_theatre_costs = theatre_costs * total_surgeries
    total_direct_system_costs = total_opd_costs + total_ipd_costs + total_theatre_costs

    return total_direct_system_costs
#=====================================================================================
# Default values for economic analyses for LF
#Productivity losses
lymphedema_params = {
    "perc_lymph_adl_pa": {"mean":.95, "lower":.90,"upper":.95},
    "adl_episodes_pa_no_mda": {"mean":4.2, "lower":2.4, "upper":9},
    "days_adl_episode": {"mean":4, "lower":1, "upper":9},
    "disability_weight_adl":{"mean":.11, "lower":.073, "upper":.157},
    "disability_weight_lymph":{"mean":.109, "lower":.073, "upper":.154},
    "perc_adl_seek_rx": {"mean":.55, "lower":.55, "upper":.7},
    "perc_lymph_visits_non_adl": {"mean":.3, "lower":.3, "upper":.55},
    "prod_loss_adl_episode": {"mean":.75, "lower":.5, "upper":.93},
    "prod_loss_chronic_lymph": {"mean":.19, "lower":.11, "upper":.31},
    "adl_reduction_mda": {"mean":.5, "lower":.15, "upper":.88},
    "lymph_reduction_mda": {"mean":.15, "lower":.01, "upper":.69}, 
}

hydrocele_params = {
    "perc_hydr_adl_pa": {"mean":.7, "lower":.45,"upper":.90},
    "adl_episodes_pa_no_mda": {"mean":4.2, "lower":2.4, "upper":9},
    "days_adl_episode": {"mean":4, "lower":1, "upper":9},
    "disability_weight_adl":{"mean":.11, "lower":.073, "upper":.157},
    "disability_weight_hydro":{"mean":.128, "lower":.086, "upper":.180},
    "perc_adl_seek_rx": {"mean":.55, "lower":.55, "upper":.7},
    "perc_hydr_visits_non_adl": {"mean":.2, "lower":.2, "upper":.55},
    "prod_loss_adl_episode": {"mean":.75, "lower":.5, "upper":.93},
    "prod_loss_chronic_hydro": {"mean":.15, "lower":.09, "upper":.24},
    "adl_reduction_mda": {"mean":.5, "lower":.15, "upper":.88},
    "hydr_reduction_mda": {"mean":.10, "lower":.01, "upper":.90}, 
    "age_hydr_sur": {"mean":40, "lower":25, "upper":50},
    "surg_success_rate": {"mean":.87, "lower":.6, "upper":.98},
    "morbidity_reduction_surg": {"mean":.90, "lower":.6, "upper":.98},
    "daly_weight_hydr": {"mean":.11, "lower":.073, "upper":.157},
}

transmission_reduction_mda = {
    "Year 1": {"Mean": .50, "std_err": .35},
    "Year 2": {"Mean": .75, "std_err": .53},
    "Year 3": {"Mean": .88, "std_err": .62},
    "Year 4": {"Mean": .94, "std_err": .66},
    "Year 5": {"Mean": .95, "std_err": .67},
    
}

#================================================================================

# terms_conditions = """
# The Taskforce for Global Health provides this tool as a service to the public. The
# Taskforce for Global Health is not responsible for, and expressly disclaims all
# liability for, damages of any kind arising out of use, reference to, or reliance on
# any information contained within this tool. While the information contained within
# this tool is periodically updated, no guarantee is given that the information
# provided in this tool is correct, complete, and up-to-date. Although this tool
# may include links providing direct access to other resources, including websites,
# The Taskforce for Global Health is not responsible for the accuracy or content of
# information contained in these sites. Links from The Taskforce for Global Health websites
# to third-party sites do not constitute an endorsement by The Taskforce of the parties
# or their products and services.

# The user agrees to indemnify the Taskforce and hold the Taskforce harmless from
# and against any and all claims, damages and liabilities asserted by third parties
# (including claims for negligence) which arise directly or indirectly from the use of
# the tool.

# """
terms_conditions = """

By using this application, you agree to be bound by the following terms and conditions:

* The application is provided as a service to the public. The Task Force for Global Health is not responsible for, and expressly disclaims all liability for, damages of any kind arising out of use, reference to, or reliance on any information contained within the application. While the information contained within the application is periodically updated, no guarantee is given that the information provided in this application is correct, complete, and up-to-date.

* The Task Force for Global Health makes no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the application or the information, products, services, or related graphics contained in the application for any purpose. Any reliance you place on such information is therefore strictly at your own risk.

* In no event will The Task Force for Global Health be liable for any loss or damage including without limitation, indirect or consequential loss or damage, or any loss or damage whatsoever arising from loss of data or profits arising out of, or in connection with, the use of this application.

* This application may include links providing direct access to other resources, including websites. The Task Force for Global Health is not responsible for the accuracy or content of information contained in these sites. Links from the application to third-party sites do not constitute an endorsement by The TaskForce for Global Health of the parties or their products and services.

* The user agrees to indemnify The Task Force for Global Health and hold it harmless from and against any and all claims, damages and liabilities asserted by third parties (including claims for negligence) which arise directly or indirectly from the use of the application.

* The Task Force for Global Health reserves the right to modify or discontinue, temporarily or permanently, the application (or any part thereof) with or without notice.

Thank you for using our application!

"""

# about_tool = """
#     This is a tool for assessing various economic scenarios for neglected tropical diseases.

#     It looks at what happens during the endgame while also looking at cumulative benefits at points in time

#     Checks for recrudescence in cases.

#     The tool relies on data from multiple sources including ESPEN, the World Bank, and CIA world factbook.
# """

about_tool = """
    To be discussed with Advocacy Team.
"""

placeholder = st.empty()

with placeholder.container():
    start_disclaimer = st.expander("Terms and Conditions")
    start_disclaimer.write(terms_conditions)
    disclaimer = st.selectbox("Click Below to Accept or Decline Terms and Conditions for Use",
    ("","Accept", "Decline"),
    index=0)
    if not disclaimer:
        st.warning("You must agree or decline the terms and conditions")
        st.stop()
    if "Decline" in disclaimer:
        st.write("""You have declined the use terms and conditions.

        Please close your browser to exit the tool.""")
        st.stop()
        
    else:
        st.write("Enter the tool")

        placeholder.empty()
#========================================================================================================================   
# This section contains code for translating the content into different languages
# #========================================================================================================================     
translator = Translator()
languages = {"English":"en", "French":"fr", "Portuguese": "pt","Spanish":"es"}
language_options = st.sidebar.selectbox("Choose Language", ("English", "French", "Portuguese", "Spanish"), index=0)
language_value = languages[language_options]

@st.cache(suppress_st_warning=True)
def translate_text(text):
    """
    Translates a dictionary of text sections from any language to the specified output language.

    Parameters:
    -----------
    sections : dict[str, str]
        A dictionary where the keys are section names and the values are the text sections to be translated.
    output_language : str, optional (default="en")
        The language to which the sections should be translated. Must be a valid ISO 639-1 language code.

    Returns:
    --------
    dict[str, str]
        A dictionary where the keys are the same as the input dictionary and the values are the translated text sections.

    Raises:
    -------
    ValueError
        If the output_language parameter is not a valid ISO 639-1 language code.

    Examples:
    ---------
    >>> sections = {"intro": "¡Hola, mundo!", "body": "Este es un ejemplo de texto para traducir."}
    >>> translated_sections = translate_sections(sections, "en")
    >>> print(translated_sections["intro"])
    Hello world!
    >>> print(translated_sections["body"])
    This is an example text to translate.
    """
    try:
        translated_text = translator.translate(text, src="en", dest=language_value).text
        return translated_text
    
    except Exception as e:
        st.write("Translation failed. Error message: " + str(e))
        return ""

def translate_markdown(markdown_text):
    """
    Translates markdown-formatted text from any language to the specified output language.

    Parameters:
    -----------
    markdown_text : str
        The markdown-formatted text to be translated.

    Returns:
    --------
    str
        The translated markdown-formatted text.
    """
    # Parse the markdown and extract any plain text

    html = markdown.markdown(markdown_text)
    plain_text = "".join(html.split("<")[0].split(">")[1:])

    # Translate the plain text
    translated_text = translate_text(plain_text)

    # Replace the plain text with the translated text
    markdown_text = markdown_text.replace(plain_text, translated_text)
    
    return markdown_text   

# #@st.cache(suppress_st_warning=True)
# def translate_cell(cell, translator):
#     """
#     Translates a single cell using Google Translate API.

#     Parameters
#     ----------
#     cell : str or int or float
#         The cell to be translated.
#     source_lang : str, optional
#         The language code of the source language. Default is 'en'.
#     target_lang : str, optional
#         The language code of the target language. Default is 'fr'.

#     Returns
#     -------
#     str or int or float
#         The translated cell. If the cell is not a string, returns the original cell.
#     """
#     if isinstance(cell, str):
#         return translator.translate(cell, src="en",dest=language_value).text
#     else:
#         return cell

# #@st.cache
# def translate_dataframe(df, src='en', dest=language_value):
#     """
#     Translates the string columns of a Pandas DataFrame from a source language to a target language.
#     Numeric columns are left unchanged.

#     Args:
#         df (pandas.DataFrame): The DataFrame to translate.
#         source_lang (str): The source language of the DataFrame. Defaults to 'en'.
#         target_lang (str): The target language to translate the DataFrame to. Defaults to 'fr'.

#     Returns:
#         pandas.DataFrame: The translated DataFrame.
#     """
#     translator = Translator()
#     translated_df = df.applymap(lambda x: translate_cell(x, translator, dest=language_value))
#     return translated_df

def translate_df(df: pd.DataFrame) -> pd.DataFrame:
    # """
    # Translates a Pandas dataframe into another language using the Google Translate API.

    # Only columns with dtype 'object' (i.e., string columns) will be translated.

    # Parameters:
    #     df (pd.DataFrame): The input dataframe to be translated.
    #     target_lang (str): The language code for the language to which the dataframe should be translated.

    # Returns:
    #     pd.DataFrame: The translated dataframe.
    # """
    translator = Translator()
    for col in df.columns:
        if df[col].dtype == 'O':
            if df[col] is not None:
                df[col] = df[col].apply(lambda x: translator.translate(x, src="en", dest=language_value).text if isinstance(x, str) and len(x.strip()) > 0 else x)
    return df

#Country inputs are various 2020 status variables from WHO and WB
#These will be used as automatic inputs once the dataset is imported
#The dataset will be converted into a dictionary for ease of manipulation
#Get country names and store them as a list and use list in a dropdown menu

@st.cache
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file from disk and returns a pandas dataframe.

    Parameters:
    -----------
    file_path: str
        The file path to the CSV file.

    Returns:
    --------
    pandas.DataFrame
        A pandas dataframe containing the contents of the CSV file.
    """
    df = pd.read_csv(file_path)
    df['Health Centre-no beds'] = df.apply(lambda row: row['Primary Hospital_OPD_Costs'] if np.isnan(row['Health Centre-no beds']) else row['Health Centre-no beds'], axis=1)
    df["Hourly_PPP(Int$)"] = df.apply(lambda row: row["Annual_PPP(Int$)"]/(row["Weekly_Work_Hours"]*52) if np.isnan(row["Hourly_PPP(Int$)"]) else row["Hourly_PPP(Int$)"], axis=1)
    df["Hourly_PPP(Int$)"] = df.apply(lambda row: row["Hourly_Nominal(USD)"] * 2.8 if row["Hourly_PPP(Int$)"] == 0 else row["Hourly_PPP(Int$)"], axis=1)
    return df

country_inputs = load_data("/Users/wobiero/Desktop/LF_Lymphasim/df_gdp.csv")
country_dict = country_inputs.set_index("Country").T.to_dict()
country_list = sorted(country_inputs["Country"].tolist())
country = st.sidebar.selectbox(
    translate_text("Select Country for Analyses"), (country_list)
)

#st.sidebar.markdown("## Menu Panel")
#===============================================================================

@st.cache
def load_country_flag(country: str) -> Image:
    """
    Loads a country flag image from disk and returns an Image object.

    Parameters:
    -----------
    country: str
        The name of the country to load the flag for.

    Returns:
    --------
    PIL.Image.Image
        An Image object representing the country flag.
    """
    file_path = "/ntd_data/datasets/flags/" + country + ".png"
    return Image.open(file_path)
country_flag = load_country_flag(country)

flag1, flag2 = st.columns([1,5], gap="medium")
with flag1:
    st.image(country_flag)
with flag2:
    st.header(translate_text(f"Economic Analysis for {country}"))

#==============================================================================


ntd_disease = st.sidebar.selectbox(
    translate_text("Target Disease"),
    ("",
    "Lymphatic filariasis",
    "Onchocerciasis",
    "Schistosomiasis",
    "Soil transmitted helminths",
    "Trachoma"), index=0, key="option"
)
if not ntd_disease:
    st.warning(translate_text("Please select NTD!"))
    st.stop()

if  "Schistosomiasis" in ntd_disease:
    st.warning(translate_text("Schistosomiasis option under construction"))
    st.stop()

if  "Trachoma" in ntd_disease:
    st.warning(translate_text("Trachoma option under construction"))
    st.stop()

if  "Soil transmitted helminths" in ntd_disease:
    st.warning(translate_text("Soil transmitted helminths option under construction"))
    st.stop()

tabs = st.tabs([translate_text("About"), 
                translate_text("Country Inputs"), 
                translate_text("Disease inputs"), 
                translate_text("Results"), 
                translate_text("Technical Assumptions"), 
                translate_text("Contact us")])

if "Lymphatic filariasis" in ntd_disease:

    with tabs[0]:
        
        st.markdown(translate_text("Please read this section before using the tool"))
        start = st.expander(translate_text("Click to read"))
        start.write(about_tool)

        # Technical notes

        lf_introductory_notes = """  

        - The default values in these tool have been sourced from public datasets including from [ESPEN](https://espen.afro.who.int/countries) and the [World Bank](https://data.worldbank.org/).  
        
        - To start, please select the relevant country for analysis in the sidebar. Then select the appropriate target NTD.    
        
        - If you want to conduct nation level analyses, select National Level under the Administration Unit 1 menu in the sidebar.          
        
        - If you want to conduct sub-national analyses, select the relevant sub-unit from the Administration Unit 1 menu, and then narrow down accordingly.
        
        - The lowest administrative unit for analysis is the Implementing Unit. 
        
        - Relevant population estimates, MDA program status, MDA geographical coverage, minimum wage rates, GDP per capita, etc are populated automatically. 
        
        - Please change the default values in the sidebar if they are incorrect or to recalibrate
        
        - If a country has failed Pre-TAS, it needs two more rounds of MDA before another Pre-TAS. Therefore set the LF status menu at MDA 4.        
        
        - The minimum time horizon to elimination is calculated as follows: 1 year for every MDA round until MDA 5, 1 year for Pre-TAS, 2 years for TAS-1 and TAS-2 ,and 3 years for TAS-3
        
        - As such, the minimum time horizon to TAS-3 for an administrative level initiating MDA will be at least 15 years.  
        
        - You can adjust the time horizon upwards (but not lower than the minimum set time horizon)
        
        - Then go to the MDA Program Details menu in the sidebar and adjust the microfilaria prevalence at the unit of analysis level. The default setting is 5%.
        
        - Cross-border endemicity refers to the LF status of geographical contiguous implementing units. This can be domestic or international.
        
        - Adjust the MDA coverage status as appropriate. If the coverage is < 60%, the MDA round is assumed to be ineffective and another year is added automatically in the analyses
        
        - The select the surveillance period at the end of TAS-3 before validation. The default is set at [6 years](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7806434/). 
        
        - The tool adjusts the time horizon by half the remaining MDA rounds if the bi-annual frequency is selected.
        
        - We assume that transmission is interrupted when the mf prevalence falls below 1%. 
        
        - Individual target programmatic costs are calculated as pop-adjusted costs. This means that by design, per capita last mile costs will be higher, although overall program costs may be lower given less drugs used
        """

        st.markdown("-----")

        lf_exp_technical = st.expander(translate_text("General notes for lymphatic filariasis module"))
        with lf_exp_technical:
            st.markdown(translate_markdown(lf_introductory_notes))
        #=====================================================================================================================================================
    with tabs[1]:
        # #=====================================================================================================================================================
        # The country_espen dataframe contains LF status details
        country_espen = pd.read_csv("/Users/wobiero/Desktop/LF_Lymphasim/espen.csv")
        country_espen.drop(["CONTINENT", "REGION", "WHO_REGION", "ADMIN0ID", "ADMIN0_FIP",
        "ADMIN1ID", "ADMIN2ID", "Alt_ADMIN2","ADMIN0ISO2", "ADMIN0ISO3", "ADMIN3ID", "IUs_ADM" ], axis=1, inplace=True)
        country_espen = country_espen[country_espen["ADMIN0"]==country] 
        mda_not_started = ["Equatorial Guinea", "Gabon"]
        mda_partial = ["Angola", "Central African Republic", "Madagascar",
        "Nigeria", "South Sudan"]
        mda_full = [
        "Burkina Faso",  "Chad", "Comoros", "Congo",
        "Côte d’Ivoire", "Democratic Republic of the Congo", "Eritrea",
        "Ethiopia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
        "Liberia", "Mali", "Mozambique", "Niger", "Senegal", "Sao Tome and Principe",
        "Sierra Leone", "United Republic of Tanzania", "Zambia", "Zimbabwe"]
        elimination_validated = ["Malawi", "Togo"]
        mda_stopped = ["Benin", "Cameroon", "Mali", "Uganda"]

        if country in elimination_validated:
            st.warning(translate_text(f"Lymphatic filariasis has been eliminated in {country}. Please select another country."))
            st.stop()
        reduce_mem_usage(country_espen)
        see_data_1 = st.expander(translate_text('Click here to see country default data'))
        with see_data_1:
            show_df(country_inputs)

        see_data_2 = st.expander(translate_text('Click here to see the raw 2020 ESPEN dataset'))
        with see_data_2:
            show_df(country_espen)
        #=====================================================================================================================================================
        st.markdown("-----")
        #=====================================================================================================================================================

        def unit_endemicity(admin_level, level):
            """
            Check the onchocerciasis endemicity status of a given administrative unit
            in a geographical area according to the ESPEN (2020) database. If the
            status is "Non-endemic", "Not reported", or "Unknown (consider Oncho
            Elimination Mapping)", issue a warning message and stop the execution.
            If the status is "Endemic (under post-intervention surveillance)",
            "Endemic (under MDA)", or a combination of both, issue an appropriate
            message.

            Parameters
            ----------
            admin_level : str
                The name of the administrative level (e.g., "IUs", "LCs", "LGAs",
                "States", etc.) used in the ESPEN database.
            level : str
                The name of the specific administrative unit for which to check the
                onchocerciasis endemicity status.

            Returns
            -------
            None
            """
            unit_status = list(set(country_espen["Endemicity"][country_espen[admin_level] == level]))
            if all("Non-endemic" == k for k in unit_status):
                st.warning(f"{level} is non-endemic for LF according to ESPEN (2020). Please select another geographical area.")
                st.stop()
            if any(k in unit_status for k in ["Endemicity unknown", "Not reported"]):
                unknown_status = list(country_espen["IUs_NAME"][(country_espen[admin_level] == level) &
                                            (country_espen["Endemicity"]=="Endemicity unknown")])
                unknown_status.extend(list(country_espen["IUs_NAME"][(country_espen[admin_level] == level) &
                                            (country_espen["Endemicity"]=="Not reported")]))
                st.warning(translate_text(f"""
                    The LF endemicity status in {', '.join(str(x) for x in unknown_status)} is unknown or not reported.
                    Contact the {level} NTD programme to get more details about these implementing units. The default
                    in this tool is that these units are non-endemic.
                    """))
                if all(k in ["Non-endemic", 'Endemic (MDA not delivered)'] for k in unit_status):
                    st.warning(translate_text(f"{level} LF programme has not begun according to ESPEN (2020)."))
                if all(k in ["Non-endemic", 'Endemic (under post-intervention surveillance)'] for k in unit_status):
                    st.warning(translate_text(f"{level} is in the post-MDA surveillance phase according to ESPEN (2020)"))
                if all("Endemic (under MDA)" == k for k in unit_status):
                    st.write(translate_text(f"{level} is under MDA"))

        # Admin 1 dropdown menu
        st.sidebar.markdown(translate_text("#### Select geographical unit for analyses."))
        admin1 = sorted(list(set(country_espen["ADMIN1"][country_espen["ADMIN0"]==country])))
        admin1.insert(0, translate_text("National Level"))
        country_admin1 = st.sidebar.selectbox(translate_text("Administrative Unit 1"), admin1)

        program_summary = st.expander("Country lymphatic filariasis status")
        with st.expander(translate_text("Geographical Unit Summary")):
            
            if translate_text("National Level") in country_admin1:
                
                pop_req_mda = (sum(country_espen["PopReq"][(country_espen["ADMIN0"]==country) &(country_espen["PopReq"]!=0)]))
                pop_tag_mda = (sum(country_espen["PopTrg"][(country_espen["ADMIN0"]==country) &(country_espen["PopTrg"]!=0)]))
                pop_trt_mda = (sum(country_espen["PopTreat"][(country_espen["ADMIN0"]==country) &(country_espen["PopTreat"]!=0)]))
                rounds_pre_tas = mda_rounds_left("ADMIN0", country, status = "LF_MDA_left")
                sub_optimal_coverage = coverage_issues(country, lower=0, upper=60)
                
                if len(sub_optimal_coverage) == 0:
                    st.write(translate_text(f"""
                    The MDA target population at the national level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country} is {rounds_pre_tas}
                    according to ESPEN (2020). Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))
                else:
                    st.write(translate_text(f"""
                    The MDA target population at the national level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country} is {rounds_pre_tas}
                    according to ESPEN (2020). Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level.

                    The following implementing units had sub-optimal MDA geographical coverage (<60%): _{';  '.join(sub_optimal_coverage).title()}_."""))
            else:
                unit_endemicity("ADMIN1", country_admin1)
                admin2 = sorted(list(set(country_espen["ADMIN2"][country_espen["ADMIN1"]==country_admin1])))
                admin2.insert(0, translate_text("Sub-National Level 1"))
                country_admin2 = st.sidebar.selectbox(translate_text("Administrative Unit 2"), admin2)

                if translate_text("Sub-National Level 1") in country_admin2:
                
                    pop_req_mda = (sum(country_espen["PopReq"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopReq"]!=0)]))
                    pop_tag_mda = (sum(country_espen["PopTrg"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopTrg"]!=0)]))
                    pop_trt_mda = (sum(country_espen["PopTreat"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopTreat"]!=0)]))
                    rounds_pre_tas = mda_rounds_left("ADMIN1", country_admin1,  status = "LF_MDA_left")
                    
                    st.write(translate_text(f"""
                    The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country_admin1.title()} is {rounds_pre_tas}
                    according to ESPEN. Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))

                if not translate_text("Sub-National Level 1") in country_admin2:
                    unit_endemicity("ADMIN2", country_admin2)
                    admin3 = sorted(list(set(country_espen["IUs_NAME"][country_espen["ADMIN2"]==country_admin2])))
                    admin3.insert(0, "Sub-National Level 2")
                    country_iu = st.sidebar.selectbox(translate_text("Implementing Unit"), admin3)

                    if translate_text("Sub-National Level 2") in country_iu:
                        
                        pop_req_mda = (sum(country_espen["PopReq"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopReq"]!=0)]))
                        pop_tag_mda = (sum(country_espen["PopTrg"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopTrg"]!=0)]))
                        pop_trt_mda = (sum(country_espen["PopTreat"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopTreat"]!=0)]))
                        rounds_pre_tas = mda_rounds_left("ADMIN2", country_admin2)

                        st.write(translate_text(f"""
                        The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                        The maximum number of MDA rounds still required in at least one implementing unit in {country_admin2.title()} is {rounds_pre_tas}
                        according to ESPEN. Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                        Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))


                    if not translate_text("Sub-National Level 2") in country_iu:
                        
                        progress_status_iu = list(country_espen["Endemicity"][country_espen["IUs_NAME"]==country_iu])
                        unit_endemicity("IUs_NAME", country_iu)
                        pop_req_mda = list(country_espen["PopReq"][country_espen["IUs_NAME"]==country_iu])[0] #Population requiring MDA
                        pop_tag_mda = list(country_espen["PopTrg"][country_espen["IUs_NAME"]==country_iu])[0] #Population targeted with MDA
                        pop_trt_mda = list(country_espen["PopTreat"][country_espen["IUs_NAME"]==country_iu])[0] #Population that received MDA
                        rounds_pre_tas = mda_rounds_left("IUs_NAME", country_iu, status = "LF_MDA_left")
                        
                        st.markdown(translate_text(f"""
                        The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                        The maximum number of MDA rounds still required in {country_iu.title()} is {rounds_pre_tas}
                        according to ESPEN. 
                        Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))

                        if 'Non-endemic' in progress_status_iu:
                            st.warning(translate_text(f"{country_iu} is non_endemic for lymphatic filariasis. Please select another IU."))
                            st.stop()
                        
        st.write("")

        #=============================================================================================

        def default_mda_stage(rounds_pre_tas):
            """
            Selects the default status for an onchocerciasis control program based on the number of previous MDA rounds.
            
            Parameters:
            -----------
            rounds_pre_tas : int
                The number of MDA rounds that have been conducted prior to transmission assessment survey (TAS).
                
            Returns:
            --------
            round_mda : int
                The default number of MDA rounds that should be conducted after the TAS.
            """
            round_mda = 0
            if country in mda_stopped:
                round_mda = 6
            if country in elimination_validated:
                round_mda = 11
            else:
                round_mda = abs(rounds_pre_tas - 5)
            return round_mda

        st.sidebar.markdown(translate_text("##### Adjust default LF programme status."))
        country_status = st.sidebar.selectbox(
            translate_text("Select LF Program Status"),
            ("No program","MDA 1", "MDA 2", "MDA 3", "MDA 4", "MDA 5", "Pre-TAS",
            "TAS 1", "TAS 2", "TAS 3", "Surveillance", "Elimination Validated"),
            index = default_mda_stage(rounds_pre_tas)
        )
        if not country_status:

            st.warning(translate_text(f"""
            In 2020, the {country} LF national program was classified by GPELF
            as {country_dict[country]['Status']}. The GPELF and ESPEN classifications may differ. 
            Please verify the current status of the programme with the country leadership before conducting analyses.
            """))
            st.stop()

        # Calculate number of years to elimination
        mda_stage = {
            "No program":14,
            "MDA 1":13,
            "MDA 2":12,
            "MDA 3":11,
            "MDA 4":10,
            "MDA 5":9,
            "Pre-TAS":8,
            "TAS 1":7,
            "TAS 2":5,
            "TAS 3":2,
            "Surveillance":0,
            "Elimination Validated":0
        }
        surv = 6 # This is set at six years post-TAS according to the literature
        min_elim_time = mda_stage[country_status] + 2020 + surv # Six added to account for surveillance
        
        # Estimate unit level MDA coverage and use that as default
        # For post-MDA countries, use 100% instead
        if pop_tag_mda == 0:
            _mda_coverage = 100
        else:
            _mda_coverage = int((pop_trt_mda/pop_tag_mda)*100)
        
        # Model predictions based on Michael et al 2004. Minimum time to elimination based on Michael E, 2004.
        predictions_header = ["coverage","endemicity_2_5", "endemicity_5_0", "endemicity_10_", "endemicity_15_"]
        coverage = [60, 70, 80, 90, 95]
        endemicity_2_5 = [7, 6, 5, 4, 3]
        endemicity_5_0 = [9, 7, 6, 5, 4]
        endemicity_10_ = [10, 8, 7, 6, 5]
        endemicity_15_ = [12, 9, 8, 7, 6]

        model_predictor = pd.DataFrame(list(zip(coverage, endemicity_2_5, endemicity_5_0, endemicity_10_, endemicity_15_)),
                            columns=predictions_header)

        if country_status:
            comp_class = st.sidebar.expander(translate_text("MDA Program Details"))
            with comp_class:
                
                st.write(translate_text(f"The at risk population is: {pop_req_mda:,}"))
                _,slider_col,_ = st.columns([.02, .96, .02])
                with slider_col:
                    # Microfilaria prevalence baseline set to zero if countries have validated elimination
                    mf_baseline = 5
                    if "mf_baseline" not in st.session_state:
                        st.session_state["mf_baseline"] = mf_baseline
                    
                    if country in elimination_validated or country in mda_stopped:
                        mf_baseline = 0
                    mf_prevalence = st.session_state["mf_prevalence"] = st.slider(translate_text("% Current microfilaria prevalence"),0,100, mf_baseline)
                    
                    mda_coverage = st.session_state["mda_coverage"] = st.slider(translate_text("% MDA coverage"),
                                            min_value = 0,
                                            max_value = 100,
                                            value= _mda_coverage)
                    if mda_coverage < 60:
                        min_elim_time += 1

                    post_surveillance_period = st.session_state["post_surveillance_period"] = st.slider(translate_text("Post surveillance time"), 0, 10)
                    # MDA coverage is only effective above 60%
                    # Add if condition for effectiveness
                    mda_frequency = st.session_state["mda_frequency"] = st.radio(translate_text("MDA Frequency"), ("Yearly", "Bi-annual"))
                    if mda_frequency == "Bi-annual":
                        min_elim_time -= 2
                    
                    # Add cross-border endemicity
                    contiguous_endemicity = st.session_state["contiguous_endemicity"] = st.radio(translate_text("Cross-border endemicity"), ("No", "Yes"))
                    if contiguous_endemicity == "Yes":
                        min_elim_time += 2
                    
                    range_selection = st.session_state["range_selection"] = st.slider(translate_text("Select time horizon in years"), 2020, 2070, min_elim_time)
                    time_horizon = range_selection - 2020 # Time horizon to be used for discounting and totals
                    
                    st.write(time_horizon)

                    st.multiselect(translate_text("Treatment Combinations"),
                    ["Ivermectin", "Albendazole", "DEC"])
                    st.write(translate_text(f"{country} treatment default is {country_dict[country]['Rx']}"))


        complementary_programs = st.sidebar.expander(translate_text("Complementary Programs"))
        with complementary_programs:
            if mf_prevalence > 5:
                st.warning(translate_text("""
                It is highly unlikely to achieve elimination with MDA alone with a mf prevalence > 5%. 
                Please verify the default status of complementary interventions.
                """))
            if mf_prevalence > 10:
                st.warning(translate_text("""
                Consider modeling the bi-annual MDA administration alongside vector control to achieve elimination
                if mf prevalence > 10%.
                """))
            _, slider_col, _ = st.columns([.02, .96,.02])
            with slider_col:
                itn_coverage = int(country_inputs["itn_2020"][country_inputs["Country"] == country])
                if itn_coverage <= 100:
                    bednets = st.slider(translate_text("% ITN coverage"), 0,100, itn_coverage)
                else:
                    bednets = st.slider(translate_text("% ITN coverage"), 0,100, 60)
                    st.write(translate_text(f"Insectide treated bednet coverage for {country} were unavailable"))
                irs = st.session_state["irs"] = st.slider(translate_text("% IRS coverage"), 0, 100, 60)
                    
        country_expander = st.sidebar.expander(translate_text("Insert country parameters"))
        with country_expander:
            econ_values()
            hh_costs_adl = st.number_input("Insert household costs ADL", value=10)
            hh_costs_lym = st.number_input("Insert household costs lymphedema", value=10)
            hh_costs_hyd = st.number_input("Insert household costs hydrocele", value=10)

        pop_expander = st.sidebar.expander(translate_text("Review unit population values"))
        # Population pyramid (%)for SSA in 5-year gaps. Over 70 combined
        ssa_pop_pyramid = [15.6, 14.0, 12.4,10.7, 9.2, 7.8,
                            6.6, 5.6, 4.5, 3.6, 2.8, 2.3, 1.7, 1.3, 1.9]
        
        with pop_expander:
            pop_distribution()

        cost_expander = st.sidebar.expander(translate_text("Insert program costs"))

        with cost_expander:
            # Remember to allocate the costs appropriately if shared
            cost_1 = translate_text("Project personnel costs")
            cost_2 = translate_text("MOH personnel costs")
            cost_3 = translate_text("Volunteer personnel costs")
            cost_4 = translate_text("Building (annualized) costs")
            cost_5 = translate_text("MDA drug costs")
            cost_6 = translate_text("Consumables costs")
            cost_7 = translate_text("MDA training costs")
            cost_8 = translate_text("TAS training costs")
            cost_9 = translate_text("Vehicle purchase (annualized) costs")
            cost_10 = translate_text("Vehicle leasing costs")
            cost_11 = translate_text("Planning meeting costs")
            cost_12 = translate_text("Mapping meeting costs")
            cost_13 = translate_text("TAS meeting costs")
            cost_14 = translate_text("Other meetings incl. conferences")
            cost_15 = translate_text("Mass drug administration costs")
            cost_16 = translate_text("Transmission assessment survey costs")
            cost_17 = translate_text("Post MDA surveillance costs")
            cost_18 = translate_text("Others e.g. per diem costs")
            
            personnel_1 = st.number_input(cost_1, value = 1000.0)
            personnel_2 = st.number_input(cost_2, value = 2000.0)
            personnel_3 = st.number_input(cost_3, value = 3000.0)

            capital_1 = st.number_input(cost_4, value = 4000.0)

            consumables_1 = st.number_input(cost_5, value = 5000.0)
            consumables_2 = st.number_input(cost_6, value = 6000.0)

            training_1 = st.number_input(cost_7, value = 7000.0)
            training_2 = st.number_input(cost_8, value = 8000.0)

            transport_1 = st.number_input(cost_9, value = 9000.0) #Remember to adjust by project share
            transport_2 = st.number_input(cost_10, value = 10000.0)

            meetings_1 = st.number_input(cost_11, value = 11000.0)
            meetings_2 = st.number_input(cost_12, value = 12000.0)
            meetings_3 = st.number_input(cost_13, value = 13000.0)
            meetings_4 = st.number_input(cost_14, value = 14000.0)

            mda_cost = st.number_input(cost_15, value=15000.0)
            tas_cost = st.number_input(cost_16, value=16000.0)
            post_mda_cost = st.number_input(cost_17, value=17000.0)

            others_1 = st.number_input(cost_18, value=18000.0)

            total_prog_cost = (personnel_1 + personnel_2 + personnel_3 + capital_1 + consumables_1 +
                consumables_2 + training_1 + training_2 + transport_1 + transport_2 +
                meetings_1 + meetings_2 + meetings_3 + meetings_4 + mda_cost +
                tas_cost + post_mda_cost + others_1)

        prog_cost_summary = {
            cost_1: personnel_1 ,
            cost_2: personnel_2 ,
            cost_3: personnel_3,
            cost_4: capital_1,
            cost_5: consumables_1,
            cost_6: consumables_2,
            cost_7: training_1,
            cost_8: training_2,
            cost_9: transport_1,
            cost_10: transport_2,
            cost_11: meetings_1,
            cost_12: meetings_2,
            cost_13: meetings_3,
            cost_14: meetings_4,
            cost_15: mda_cost,
            cost_16: tas_cost,
            cost_17: post_mda_cost,
            cost_18: others_1,
            "Total": total_prog_cost
        }

        prog_cost_df = pd.DataFrame(prog_cost_summary.items(), columns=["Category", "Estimate"])
        prog_cost_expander = st.expander(translate_text("Summary of Progammatic Costs"))
        prog_cost_expander.dataframe(prog_cost_df)
        
        patient_costs = st.sidebar.expander(translate_text("Average patient costs"))
        with patient_costs:
            medication_costs = st.number_input(translate_text("Drug costs for OPD visit"), value = 1)
            laboratory_costs = st.number_input(translate_text("Laboratory costs for OPD visit"), value=1)
            surgical_costs = st.number_input(translate_text("Surgical costs for hydrocele"), value=1) 
            inpatient_costs = st.number_input(translate_text("Inpatient costs for hydrocele surgery"), value=1)
            dressing_costs = st.number_input(translate_text("Limb care (dressing) costs"), value=1)
            consultation_costs = st.number_input(translate_text("OPD consultation costs"), value=1)
            travel_costs = st.number_input(translate_text("Average travel costs"), value=1)
            avg_time_at_clinic = st.number_input(translate_text("Average time spent at facility in hours"), value=1)
            avg_los_surgery = st.number_input(translate_text("Average length of stay hydrocele surgery in days"), value=1)
            time_costs_adl = avg_time_at_clinic * daily_wage()[1]
            time_costs_hydro = avg_los_surgery * daily_wage()[0]
            patient_clinical_adl_costs = (
                medication_costs + laboratory_costs + consultation_costs + travel_costs
            )
            patient_non_clinical_adl_costs = medication_costs
            patient_hydrocele_costs = (
                medication_costs + laboratory_costs + consultation_costs + travel_costs +
                inpatient_costs + surgical_costs + dressing_costs
            )
            medical_cost_inflation = (3 + country_inputs['Inflation rate (consumer prices) (%)'][country_inputs["Country"]==country])/100

        technical_expander = st.sidebar.expander(translate_text("Technical parameters (In-built defaults))"))
        with technical_expander:

            # Disability weights from the GBD study
            dw_lymphedema = st.number_input(translate_text("Disability weight lymphedema(default=.105)"), value=.105)
            dw_hydrocele = st.number_input(translate_text("Disability weight hydrocele(default=.073)"), value=.073)
            avg_onset = st.number_input(translate_text("Age of symptom onset LF(default=20)"), value=20)
            #Economic defaults
            disc_costs = st.number_input(translate_text("Discount rate costs"), value=0.03)
            disc_effects = st.number_input(translate_text("Discount rate effects"), value=0.03)

            # Health system utilization
            hyd_visits = st.number_input(translate_text("Hydrocele clinic visits p.a."), value=2)
            lym_visits = st.number_input(translate_text("Lymphedema clinic visits p.a."), value=4)
            adl_visits = st.number_input(translate_text("ADL clinic visits p.a."), value=4.65)
            adl_episode = st.number_input(translate_text("Duration of ADL episode"), value=3.93)

            # Productivity loses
            prod_loss_hyd = st.number_input(translate_text("Productivity loss hydrocele"), value=.15)
            prod_loss_lym = st.number_input(translate_text("Productivity loss lymphedema"), value=.18)
            prod_loss_adl = st.number_input(translate_text("Productivity loss ADL"), value=.78)

            # Other parameters:
            at_risk_pop = st.number_input(translate_text("At risk population(default=10%)"), value=pop_req_mda)
            lf_to_lymph = st.number_input(translate_text("% of LF infections->lymphedema(default=12.5%)"), value=.125)
            lf_to_hydro = st.number_input(translate_text("% of LF infections->hydrocele(default=20.8%)"), value=.208)
            lf_to_subcl = st.number_input(translate_text("% of LF infections subclinical(default=66.7%)"), value=.667)
            daly_av_hydr = st.number_input(translate_text("DALY Averted per prevented hydrocele(default=2.3)"), value=2.3)
            daly_av_lymh = st.number_input(translate_text("DALY averted per prevented lymphedema(default=3.3)"), value=3.3)

        baseline_inputs = st.expander(translate_text("Estimated baseline population figures"))
        with baseline_inputs:  
            lymphedema_cases = caseloads(at_risk_pop, time_horizon)[1]
            adl_cases_total = hydrocele_cases = caseloads(at_risk_pop, time_horizon)[2]
            hydrocele_cases = caseloads(at_risk_pop, time_horizon)[0]

            st.write(translate_text(f"""We estimate {hydrocele_cases:,.0f} chronic hydrocele cases, {lymphedema_cases:,.0f} chronic 
                        lymphedema cases, and {adl_cases_total:,.0f} annual ADL cases at this level in the counterfactual scenario of no MDA."""))
            
            lymph_df = pd.read_csv("/Users/wobiero/Desktop/LF_Lymphasim/lymph_proportions.csv")
            lymph_df["Cases"] = [x * lymphedema_cases for x in lymph_df["First"]]
            stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6", "Stage 7"]
            
            cohort_name = ["10-14","15-19", '20-24', "25-29", "30-34", "35-39", "40-44", 
                "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
            lymph_df[stages] = lymph_df[stages].multiply(lymph_df["Cases"], axis="index")
            lymph_df.drop(["Cases", "Age Cohort"], axis=1, inplace=True)
            lymph_df = lymph_df.T
            lymph_df.columns = cohort_name
            
            lymph_df = lymph_df[1:]
            
            lymph_df["stages"] = stages

            lymph_figure_1 = alt.Chart(lymph_df.melt("stages")).mark_line(point=True).encode(
                x=alt.X("stages", title=" "),
                y=alt.Y("value", title="Estimated patients"),
                color="variable"
            ).properties(
                width=400,
                height=450
            )
            
            lymphedema_dict = {
                "Stage 1": int(lymphedema_cases * .395),
                "Stage 2": int(lymphedema_cases * .286),
                "Stage 3": int(lymphedema_cases * .219),
                "Stage 4": int(lymphedema_cases * .066),
                "Stage 5": int(lymphedema_cases * .016),
                "Stage 6": int(lymphedema_cases * .011),
                "Stage 7": int(lymphedema_cases * .007)
            }
            lymphedema_table = pd.DataFrame(lymphedema_dict.items(), columns=["Stage", "Estimates"])

            lymphedema_narrative = translate_text(f"""
            There are approximately {lymphedema_cases:,.0f} lymphedema cases at the selected administrative level.
            Using proportions in [Sawers L and Stillwagon E, (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7356420/), the distribution by severity is
            shown in the table below. The age-disaggregated caseload by severity is shown in the adjacent figure. 
            """)

        country_maps = st.expander(translate_text(f"Click here to see and download the {country} MDA programme status map"))
        with country_maps:
            # @st.cache(suppress_st_warning=True)
            # def load_country_map(country: str) -> Image:
            #     """
            #     Loads a country map image from disk and displays it along with a download button.

            #     Parameters:
            #         country (str): The name of the country for which to load the map.

            #     Returns:
            #         tuple: A tuple containing the loaded image object and a download button object.
            #             If the image file is not found, the function returns None.

            #     Raises:
            #         None
            #     """
            #     map_path = os.path.join(os.getcwd(), "LF_Lymphasim/country_maps", f"{country}.png")
            #     try:
            #         country_map = Image.open(map_path)
            #         st.image(country_map)
            #         download_button = st.download_button(label = f"Download {country} map",
            #                         data = open(map_path, 'rb').read(),
            #                         file_name =country+".png",
            #                         mime = "image/png")
            #         return country_map, download_button
            #     except IOError:
            #         st.write(f"Static map for {country} not found. Please check source directory.")
            try:
                country_map_1 = Image.open("/Users/wobiero/Desktop/LF_Lymphasim/country_maps/"+country+".png")
                st.image(country_map_1)
                st.download_button(label = f"Download {country} map",
                                    data = open("/Users/wobiero/Desktop/LF_Lymphasim/country_maps/"+country+".png", 'rb').read(),
                                    file_name =country+".png",
                                    mime = "image/png")
            except IOError:
                st.write(translate_text(f"Static map for {country} not found. Please check source directory."))

        interactive_map = st.expander(translate_text(f"Click here if you want to see the interactive map for {country}"))
        with interactive_map:
            # @st.cache
            # def load_html_map(country: str) -> str:
            #     """
            #     Loads an HTML map file from disk.

            #     Parameters:
            #         country (str): The name of the country for which to load the map.

            #     Returns:
            #         str: The contents of the HTML file.

            #     Raises:
            #         IOError: If the HTML file is not found.
            #     """
            #     html_map_path = "/Users/wobiero/Desktop/LF_Lymphasim/maps_html/" + country + ".html"
            #     try:
            #         with open(html_map_path, "r", encoding="utf-8") as f:
            #             return f.read()
            #     except IOError:
            #         raise IOError(f"HTML map for {country} not found. Please check source directory.")
                
            # # get the HTML map for the selected country
            # html_map = load_html_map(country)

            # # display the HTML map using st.components.html()
            # components.html(html_map, width=1000, height=1500)


            try:
                html_map = open("/Users/wobiero/Desktop/LF_Lymphasim/maps_html/"+country+".html", "r", encoding='utf-8')
                source_code =html_map.read()
                components.html(source_code, width=1000, height=1500)
            except IOError:
                st.write(translate_text(f"HTML map for {country} not found. Please check source directory."))

        @dataclass
        class ModelInputs:
            n_iterations: int = 1000
        
            hydrocele_cases: float = hydrocele_cases
            hydrocele_cases_std: float = hydrocele_cases * .1
            
            lymphedema_cases: float = lymphedema_cases
            lymphedema_cases_std: float = lymphedema_cases * .1
            
            perc_lymph_adl_pa: float = .95
            perc_lymph_adl_pa_std: float = .025

            adl_episodes_pa_no_mda: float = 4.2
            adl_episodes_pa_no_mda_std: float = 1.7

            days_adl_episode: float = 4.0
            days_adl_episode_std: float = 2.0

            disability_weight_adl: float = .11
            disability_weight_adl_std: float = .021

            disability_weight_lymph: float = .109
            disability_weight_lymph_std: float = .020

            perc_adl_seek_rx: float = .55
            perc_adl_seek_rx_std: float = .04

            perc_lymph_visits_non_adl: float = .3
            perc_lymph_visits_non_adl_std: float = .06

            prod_loss_adl_episode: float = .75
            prod_loss_adl_episode_std: float = .11

            prod_loss_chronic_lymph: float = .19
            prod_loss_chronic_lymph_std: float = .05

            adl_reduction_mda: float = .5
            adl_reduction_mda_std: float = .18

            lymph_reduction_mda: float = .15
            lymph_reduction_mda_std: float = .15

            perc_hydr_adl_pa: float = .7
            perc_hydr_adl_pa_std: float = .11

            adl_episodes_pa_no_mda: float = 4.2
            adl_episodes_pa_no_mda_std: float = 1.65

            disability_weight_hydro: float = .128
            disability_weight_hydro_std: float = .023

            perc_hydr_visits_non_adl: float = .2
            perc_hydr_visits_non_adl_std: float = .08

            prod_loss_chronic_hydro: float = .15
            prod_loss_chronic_hydro_std: float = .04

            hydr_reduction_mda: float = .10
            hydr_reduction_mda_std: float = .23

            age_hydr_sur: float = 40.0
            age_hydr_sur_std: float = 6.25

            surg_success_rate: float = .87
            surg_success_rate_std: float = .23

            morbidity_reduction_surg: float = .90
            morbidity_reduction_surg_std:  float = .23

            daly_weight_hydr: float = .11
            daly_weight_hydr_std: float = .21
            
            clinic_consultation_time: float = 0.0625 # We assume an 8 hour work-day
            
            surgical_length_of_stay: float = 2.0 # We assume two days in hospital -- prep, labs, surgery, discharge
            surgical_length_of_stay_std: float = .5
            
            surgical_review_meetings: float = 2.5 # Assume booking, post-op reviews, consultations
            surgical_review_meetings_std: float = .3
                
            hydr_surgeries_perc: float = .5 # Proportion of hydrocele cases that undergo surgeries in a lifetime
            hydr_surgeries_perc_std: float = .1
            
            hydr_prop_pa: float = .05 # Percentage of hydrocele surgical cases that are done annually
            hydr_prop_pa_std: float = .03
            
            hosp_stay_surg: float = 2.0
            hosp_stay_surg_std: float = 1.0
            
            operation_time: float = 0.125
            operation_time_std: float = 0.025
            
            prop_dressed: float = .1 # proportion hydrocele dressed in community
            prop_dressed_std: float = .01
            
            clinic_consultation_time: float = 0.0625 # Annual inpatient days
            
            r : float = .015
            r_std: float = .005
    
        sim_data = ModelInputs()  
    with tabs[2]:

        lymphedema_details = st.expander(translate_text("Lymphedema Population Estimates for Unit"))
        with lymphedema_details:
            st.write(lymphedema_narrative)
            col1, col2 = st.columns(2)
            with col1:
                st.write(translate_text("Estimated lymphedema cases by stage"))
                st.table(lymphedema_table)
                st.write(translate_text(f"Total cases: {lymphedema_cases:,.0f}"))
            with col2:
                st.write(lymph_figure_1)

            st.write(translate_text("""
            Please note that Stages 1 and 2 only show slight lower limb swellings which patients often ignore. 
            Following [Sawers L and Stillwagon E, (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7356420/), we assume that patients in Stages 1 and 2 do not suffer any productivity losses from
            lymphedema, while those in Stage 7 are severely disabled and are not economically productive. We also assume that MDA does not resolve
            lymphedema but prevents disease progression. All patients with lymphedema still suffer ADL episodes in this model. The detailed breakdown is shown in the table below.
            """))    
            st.table(lymph_df)

        hydrocele_details = st.expander(translate_text("Hydrocele Population Estimates for Unit"))
        with hydrocele_details:
            st.write(translate_text(f"""
            We expect {hydrocele_cases:,.0f} at this administrative level. We assume that half the cases - {hydrocele_cases/2:,.0f} - will undergo surgery in
            their lifetime with a 87% success rate, and 90% morbidity reduction. The decline in demand for surgery follows a logistic pattern with initial high demand and a longer tail with low demand.
            The initial high demand is assumed to stem from higher surgical interventions for the larger hydroceles.
            """))
            
            hydrocele_declines = hydrocele_plotter(sim_data)
            st.write(hydrocele_declines)

    with tabs[3]:
        simulated_df = monte_carlo_data(sim_data)
        #------------------------------------------------------------------------------------------------------------
        # Non-MDA scenario
        days_lost = annual_workdays_lost_no_mda(simulated_df,lymphedema_cases, hydrocele_cases)[0]
        days_narrative = annual_workdays_lost_no_mda(simulated_df,lymphedema_cases, hydrocele_cases)[1]
        dalys_lost = dalys_pa(hydrocele_cases, lymphedema_cases, simulated_df)

        indicators = ["Annual Costs", "Elimination Date", 
                    "Direct Health Sector Costs", "Out-Of-Pocket Costs", "Consultation Hours",
                    "DALYs",
                    "Economic Workdays Lost", "Lost Labor Costs", "Total Economic Costs", ]
        values_non_disc = np.repeat(1, len(indicators))
        values_disc = np.repeat(1, len(indicators))
        
        # We assume that OOP is approximately 4 hours of lost time per visit.
        simulated_df["OOP"] = (simulated_df["annual_adl_days"] + simulated_df["annual_non_adl_days"]) * 24 * econ_values()[0] * 3
        lf_oop_costs = simulated_df["OOP"].mean()
        labor_loss = (days_lost.iloc[-1]["Mean"] * daily_wage()[0]).tolist()
        health_sector_hours = simulated_df['annual_health_sector_time'].mean() * 8
        health_sector_costs = direct_health_sector_costs().tolist()[0]
        prog_costs = prog_cost_df.loc[prog_cost_df["Category"]=="Total", "Estimate"].iloc[0] 

        tot_lf_econ_costs = health_sector_costs + prog_costs + lf_oop_costs + labor_loss
        
        econ_results = pd.DataFrame(list(zip(indicators, values_non_disc, values_disc)),
                            columns=["Indicator", "Discounted Values", "Cumulative-Discounted Values"])
    
        econ_results.loc[econ_results["Indicator"]=="Elimination Date", "Discounted Values"] = 2020 + time_horizon
        econ_results.loc[econ_results["Indicator"]=="Elimination Date", "Cumulative-Discounted Values"] = 2020 + time_horizon

        #econ_results.loc[econ_results["Indicator"]=="Baseline Population (at risk)", "Discounted Values"] = '{:,.0f}'.format(at_risk_pop)
        #econ_results.loc[econ_results["Indicator"]=="Baseline Population (at risk)", "Cumulative-Discounted Values"] = '{:,.0f}'.format(at_risk_pop)

        econ_results.loc[econ_results["Indicator"]=="Economic Workdays Lost", "Discounted Values"] = '{:,.0f}'.format(days_lost.iloc[-1]["Mean"])
        econ_results.loc[econ_results["Indicator"]=="Economic Workdays Lost", "Cumulative-Discounted Values"] = '{:,.0f}'.format(cost_discounter([days_lost.iloc[-1]["Mean"]]*time_horizon, time_horizon, disc_costs=disc_costs))
        
        econ_results.loc[econ_results["Indicator"]=="DALYs", "Discounted Values"] = '{:,.0f}'.format(dalys_lost.iloc[-1]["Mean"])
        econ_results.loc[econ_results["Indicator"]=="DALYs", "Cumulative-Discounted Values"] = '{:,.0f}'.format(cost_discounter([dalys_lost.iloc[-1]["Mean"]]*time_horizon, time_horizon, disc_costs=disc_costs))

        econ_results.loc[econ_results["Indicator"]=="Lost Labor Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(labor_loss[0]))
        econ_results.loc[econ_results["Indicator"]=="Lost Labor Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(labor_loss[0]))

        econ_results.loc[econ_results["Indicator"]=="Consultation Hours", "Discounted Values"] = '{:,.0f}'.format(health_sector_hours)
        econ_results.loc[econ_results["Indicator"]=="Consultation Hours", "Cumulative-Discounted Values"] = '{:,.0f}'.format(health_sector_hours * time_horizon)

        econ_results.loc[econ_results["Indicator"]=="Direct Health Sector Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(health_sector_costs))
        econ_results.loc[econ_results["Indicator"]=="Direct Health Sector Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([health_sector_costs]*time_horizon, time_horizon, disc_costs=disc_costs)))
        
        econ_results.loc[econ_results["Indicator"]=="Out-Of-Pocket Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(lf_oop_costs))
        econ_results.loc[econ_results["Indicator"]=="Out-Of-Pocket Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([lf_oop_costs]*time_horizon, time_horizon, disc_costs=disc_costs)))       
        
        econ_results.loc[econ_results["Indicator"]=="Annual Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(prog_costs *.13))
        econ_results.loc[econ_results["Indicator"]=="Annual Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([prog_costs * .13]*time_horizon, time_horizon, disc_costs=disc_costs)))  

        econ_results.loc[econ_results["Indicator"]=="Total Economic Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(tot_lf_econ_costs[0]))
        econ_results.loc[econ_results["Indicator"]=="Total Economic Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([tot_lf_econ_costs[0]]*time_horizon, time_horizon, disc_costs=disc_costs))) 

        see_econ_results = st.expander(translate_text("Click to see summary of results - no MDA scenario"))
        
        with see_econ_results:
            show_df(econ_results)

            nmhsht = hydrocele_clinic_time_no_mda(simulated_df) 
            st.write(translate_text(f"""
            In a non-MDA scenario, approximately {nmhsht[4]:,.0f} workdays would have been spent on hydrocele
            management annually within the health sector at this administrative level. This includes approximately {nmhsht[1]:,.0f} surgeon days
            for hydrocelectomies assuming there were resources to perform them. These calculations assume an 8-hour workday for elective cases.
            The assumptions around surgical need are based on population projections and need to be verified with the MOH leadership. 
            Look at the Monte Carlo simulations table in the preceding tab.
            """))

            st.dataframe(styler(days_lost))
            st.write(days_narrative)               
            st.dataframe(styler(dalys_lost))
                    
        # Lymphedema technical inputs
        a = ['perc_lymph_adl_pa', 0.95, 0.9, 0.95]
        b = ['adl_episodes_pa_no_mda', 4.2, 2.4, 9]
        c = ['days_adl_episode', 4, 1, 9]
        d = ['disability_weight_adl', 0.11, 0.073, 0.157]
        e = ['disability_weight_lymph', 0.109, 0.073, 0.154]
        f = ['perc_adl_seek_rx', 0.55, 0.55, 0.7]
        g = ['perc_lymph_visits_non_adl', 0.3, 0.3, 0.55]
        h = ['prod_loss_adl_episode', 0.75, 0.5, 0.93]
        i = ['prod_loss_chronic_lymph', 0.19, 0.11, 0.31]
        j = ['adl_reduction_mda', 0.5, 0.15, 0.88]
        k = ['lymph_reduction_mda', 0.15, 0.01, 0.69]
        l = [a, b, c, d, e, f, g, h, i, j, k]

        lym_df_technical = pd.DataFrame(l, columns = ["parameter", "mean", "min", "max"])
        
        # Hydrocele technical inputs

        a1 = ['perc_hydr_adl_pa', 0.7, 0.45, 0.9]
        a2 = ['adl_episodes_pa_no_mda', 2.1, 0.4, 7.0]
        a3 = ['days_adl_episode', 4, 1, 9]
        a4 = ['disability_weight_adl', 0.11, 0.073, 0.157]
        a5 = ['disability_weight_hydro', 0.128, 0.086, 0.18]
        a6 = ['perc_adl_seek_rx', 0.55, 0.55, 0.7]
        a7 = ['perc_hydr_visits_non_adl', 0.2, 0.2, 0.55]
        a8 = ['prod_loss_adl_episode', 0.75, 0.5, 0.93]
        a9 = ['prod_loss_chronic_hydro', 0.15, 0.09, 0.24]
        a10 = ['adl_reduction_mda', 0.5, 0.15, 0.88]
        a11 = ['hydr_reduction_mda', 0.1, 0.01, 0.9]
        a12 = ['age_hydr_sur', 40, 25, 50]
        a13 = ['surg_success_rate', 0.87, 0.6, 0.98]
        a14 = ['morbidity_reduction_surg', 0.9, 0.6, 0.98]
        a15 = ['daly_weight_hydr', 0.11, 0.073, 0.157]
        a16 = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                a11, a12, a13, a14, a15]
        hydr_df_technical = pd.DataFrame(a16, columns = ["parameter", "mean", "min", "max"])


    #------------------------------------------------------------------------------------------------------------
    # MDA scenario
    #------------------------------------------------------------------------------------------------------------ 
        days_lost_mda = annual_workdays_lost_with_mda(simulated_df,lymphedema_cases, hydrocele_cases)[0]
        days_narrative_mda = annual_workdays_lost_with_mda(simulated_df,lymphedema_cases, hydrocele_cases)[1]
        dalys_lost_mda = dalys_pa_mda(hydrocele_cases, lymphedema_cases, simulated_df)
        
        labor_loss_mda = (days_lost_mda.iloc[-1]["Mean"] * daily_wage()[0]).tolist()
        health_sector_hours_mda = (simulated_df['annual_health_sector_time'] * 8 *
                                    simulated_df["adl_reduction_mda"] * simulated_df["hydr_reduction_mda"]
                                ).mean()
        health_sector_costs_mda = (direct_health_sector_costs().tolist()[0] *
                                    simulated_df["adl_reduction_mda"] * simulated_df["hydr_reduction_mda"]
                                ).mean()
    
        lf_oop_costs_mda = (simulated_df["adl_reduction_mda"] * simulated_df["hydr_reduction_mda"] * simulated_df["OOP"]).mean()
        
        tot_lf_econ_costs_mda = health_sector_costs_mda + prog_costs + lf_oop_costs_mda + labor_loss_mda

        econ_results_mda = pd.DataFrame(list(zip(indicators, values_non_disc, values_disc)),
                            columns=["Indicator", "Discounted Values", "Cumulative-Discounted Values"])
    
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Elimination Date", "Discounted Values"] = 2020 + time_horizon
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Elimination Date", "Cumulative-Discounted Values"] = 2020 + time_horizon
        
        # econ_results_mda.loc[econ_results_mda["Indicator"]=="Baseline Population (at risk)", "Discounted Values"] = '{:,.0f}'.format(at_risk_pop)
        # econ_results_mda.loc[econ_results_mda["Indicator"]=="Baseline Population (at risk)", "Cumulative-Discounted Values"] = '{:,.0f}'.format(at_risk_pop)
        
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Economic Workdays Lost", "Cumulative-Discounted Values"] = '{:,.0f}'.format(days_lost_mda.iloc[-1]["Mean"])
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Economic Workdays Lost", "Discounted Values"] = '{:,.0f}'.format(cost_discounter([days_lost_mda.iloc[-1]["Mean"]]*time_horizon, time_horizon, disc_costs=disc_costs))
        
        econ_results_mda.loc[econ_results_mda["Indicator"]=="DALYs", "Discounted Values"] = '{:,.0f}'.format(dalys_lost_mda.iloc[-1]["Mean"])
        econ_results_mda.loc[econ_results_mda["Indicator"]=="DALYs", "Cumulative-Discounted Values"] = '{:,.0f}'.format(cost_discounter([dalys_lost_mda.iloc[-1]["Mean"]]*time_horizon, time_horizon, disc_costs=disc_costs))

        econ_results_mda.loc[econ_results_mda["Indicator"]=="Lost Labor Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(labor_loss_mda[0]))
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Lost Labor Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([labor_loss_mda[0]]*time_horizon, time_horizon, disc_costs=disc_costs)))

        econ_results_mda.loc[econ_results_mda["Indicator"]=="Consultation Hours", "Discounted Values"] = '{:,.0f}'.format(health_sector_hours_mda)
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Consultation Hours", "Cumulative-Discounted Values"] = '{:,.0f}'.format(health_sector_hours_mda * time_horizon)

        econ_results_mda.loc[econ_results_mda["Indicator"]=="Direct Health Sector Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(health_sector_costs_mda))
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Direct Health Sector Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([health_sector_costs_mda]*time_horizon, time_horizon, disc_costs=disc_costs)))
        
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Out-Of-Pocket Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(lf_oop_costs_mda))
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Out-Of-Pocket Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([lf_oop_costs_mda]*time_horizon, time_horizon, disc_costs=disc_costs)))   

        econ_results_mda.loc[econ_results_mda["Indicator"]=="Annual Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(prog_costs))
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Annual Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([prog_costs]*time_horizon, time_horizon, disc_costs=disc_costs))) 

        econ_results_mda.loc[econ_results_mda["Indicator"]=="Total Economic Costs", "Discounted Values"] = "$" + str('{:,.2f}'.format(tot_lf_econ_costs_mda[0]))
        econ_results_mda.loc[econ_results_mda["Indicator"]=="Total Economic Costs", "Cumulative-Discounted Values"] = "$" + str('{:,.2f}'.format(cost_discounter([tot_lf_econ_costs_mda[0]]*time_horizon, time_horizon, disc_costs=disc_costs))) 
        
        see_econ_results_mda = st.expander(translate_text("Click to see summary of results - MDA scenario"))
        

        with see_econ_results_mda:
            show_df(econ_results_mda)

            mhsht = hydrocele_clinic_time_mda(simulated_df)  # mda_health_sector_hydrocele_time
            st.write(translate_text(f"""
                In a MDA scenario, approximately {mhsht[4]:,.0f} workdays would have been spent on hydrocele
                management annually within the health sector at this administrative level. This includes approximately {mhsht[1]:,.0f} surgeon days
                for hydrocelectomies assuming there were resources to perform them. These calculations assume an 8-hour workday for elective cases.
                The assumptions around surgical need are based on population projections and need to be verified with the MOH leadership. 
                Look at the Monte Carlo simulations table in the preceding tab.
            """))

            st.dataframe(styler(days_lost_mda))
            st.write(days_narrative_mda)
            st.dataframe(styler(dalys_lost_mda))

        lf_nmb = st.expander(translate_text("Lymphatic filariasis MDA economic returns"))
        with lf_nmb:
            econ_diff_pa = tot_lf_econ_costs[0] - tot_lf_econ_costs_mda[0]
            prog_costs_diff = prog_costs * .87
            nmb_lf = econ_diff_pa/prog_costs_diff
            gdp_ppp_x = country_inputs["Annual_PPP(Int$)"][country_inputs["Country"]==country].values[0]
            qaly_diff_lf = 1 - (dalys_lost_mda.iloc[-1]["Mean"] - dalys_lost.iloc[-1]["Mean"]) # QALY change LF
            qaly_diff_lf_l = 1 -(dalys_lost_mda.iat[3,2] - dalys_lost.iat[3,2])
            qaly_diff_lf_u = 1 -(dalys_lost_mda.iat[3,3] - dalys_lost.iat[3,3])
            cea_lf_qaly = prog_costs_diff/qaly_diff_lf #ICER LF
            
            st.write(translate_text(f"""For every USD invested in LF MDA programs, there is an economic return of {nmb_lf:,.2f} USD for every dollar invested. In this ROI analysis, we do not look at spillover benefits on other diseases that
                can be controlled or treated with the same MDA regimens like soil-transmitted helminths. This estimate should therefore be viewed as a conservative estimate. The incremental costs per QALY is {cea_lf_qaly:,.2f}. 
                This value is cost effective whether one uses the somewhat controversial GDP-based threshold of {gdp_ppp_x:,.2f} USD per QALY or the alternative threshold of {cea_threshold():,.2f} USD per QALY using the approach 
                by [Woods et al (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5193154/)."""))
            st.write('*<p style="color:red;">Note: we use placeholding values for program costs that need to be adjusted using country specific data.*', unsafe_allow_html=True)

        detailed_results = st.expander(translate_text("Monte Carlo Simulations Table. Click to see and download."))

        with detailed_results:
            st.dataframe(simulated_df.style.format("{:,.3f}"))
            st.download_button(translate_text("Download Summary"), simulated_df.to_csv(), mime="text/csv")
    with tabs[4]:
        
        lymphedema_inputs = st.expander(translate_text("Lymphedema technical inputs"))
        with lymphedema_inputs:
            st.table(lym_df_technical)

        hydrocele_inputs = st.expander(translate_text("Hydrocele technical inputs"))
        with hydrocele_inputs:
            st.table(hydr_df_technical)
        
        # A summary of economic parameters used in the tool

        economic_parameters = ["Discount rate", "Per capita GDP", "Daily productivity ($)",
                                "Medical cost inflation", "Weekly work hours", "Inequality*","Life expectancy",
                                "Upper CEA Threshold", "Time horizon"]

        economic_parameter_values = np.repeat(1, len(economic_parameters))

        economic_parameters_df = pd.DataFrame(list(zip(economic_parameters, economic_parameter_values)),
                                    columns=["Parameter", "Value"])
    
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Discount rate", "Value"] = float(disc_costs)
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Inequality*", "Value"] = round(float(country_inputs["inequality_.2_quintile"][country_inputs["Country"]==country]/100),2)
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Daily productivity ($)", "Value"] = round(float(daily_wage()[0]),2)
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Life expectancy", "Value"] = float(country_inputs["Life_Expectancy"][country_inputs["Country"]==country])
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Weekly work hours", "Value"] = float(country_inputs["Weekly_Work_Hours"][country_inputs["Country"]==country])
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Per capita GDP", "Value"] = float(country_inputs["cia_estimate"][country_inputs["Country"]==country])
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Medical cost inflation", "Value"] = round(float(medical_cost_inflation),2)
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Upper CEA Threshold", "Value"] = float(country_inputs["upper_ppp_cet"][country_inputs["Country"]==country])
        economic_parameters_df.loc[economic_parameters_df["Parameter"]=="Time horizon", "Value"] = float(time_horizon)
        see_econ_parameters = st.expander(translate_text("Click to see summary of economic model inputs"))

        with see_econ_parameters:
            show_df(economic_parameters_df)
            st.caption(translate_text("""
            The daily productivity value is minimum daily wage adjusted by the proportion of wealth held by the bottom quintile (inequality*) and number
            of working days per year. This adjustment is done to reflect the inequalities in distribution of LF. That is, this incorporates a form of 
            "quintile dispersion ratio" as an equity measure in the tool. See [Mathew et al, (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7286370/) for a detailed explanation. 
            """))
        st.markdown("------------")

        # Technical notes
        lf_technical_notes = """

        -   The default values in this tool are from various sources. 

        -   You can change the default values if you have better/local data 

        -   Albendazole + Ivermectin (Onchocerciasis endemic areas) 

        -   Elimination target microfilarial prevalence < 1% [Anopheles/Culex dominant] 

        -   Elimination target microfilarial prevalence <.5% [Aedes dominant] 

        -   Antigenemia prevalence 2% as proxy for microfilarial prevalence
        -   WHO guidelines 5 annual rounds of MDA before pre-transmission assessment survey
        -   TAS-1 after pre-TAS. MDA stopped if TAS-1 passed
        -   TAS-2 and TAS-3 have to be passed before elimination is certified
        -   TAS-2 and TAS-3 conducted each within 2-3 years of preceding TAS
        -   Possibility of ongoing transmission despite passing TAS
        -   Assumption no reinfection
        -   We do not look at spillover effects across NTDs under this tab (see combo tab)
        -   Recommended post surveillance period 4 - 6 years
        -   Risk of recrudesence
        -   % Microfilaraemia clearance: DEC (90%), Ivermectin(99%), DEC + ABZ (95%), Ivermectin + ABZ (99%)
        -   % Adult worm killed: DE (35%), Ivermectin(11%), DEC+ABZ(55%), Ivermectin+ABZ(35%)
        -   Reduction in months Microfilaraemia: DEC(3), Ivermectin(9), DEC+ABZ(6), Ivermectin+ABZ(9)
        -   Source: [Michael et al, (2004)](https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(04)00973-9/fulltext)
        -   We assume that medical cost inflation is on average 3 percentage points higher than general inflation
        -   % workdays lost lymphedema Stages 1 & 2 - 0, Stage 3 - 20%, Stage 4 - 50%, Stages 5 & 6 - 75%, Stage 7 - 100%
        -   Since hydrocele prevalence increases by age, we assume an average age of 30 years for the hydrocele analyses
        -   The tool automatically adds two years of extra surveillance if a contiguous geographical unit is still endemic for LF
        -   While Cameroon is classified generally as post-mda surveillance, one IU, Akwaya in Sud Ouest is classified by ESPEN as requiring
            MDA. This tool assumes that the entire country is not post-MDA if at least one IU is still under MDA. The national program should clarify the correct status.

        """

        lf_exp_technical = st.expander(translate_text("Technical Notes for the Lymphatic Filariasis Models. These can be adjusted in the sidebar if necessary."))
        with lf_exp_technical:
            st.markdown(translate_markdown(lf_technical_notes))
        st.markdown("-----")

    
    with tabs[5]:
        with st.expander("Contact us"):
            contact_form = """
                <form action="https://formsubmit.co/obierochieng@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder="Your name" required>
                    <input type="email" name="email" placeholder="Your email" required>
                    <input type="text" name="_honey" style="display:none">
                    <input type="hidden" name="_cc" value="ocu9@cdc.gov">
                    <textarea name="message" placeholder="Details of your problem"></textarea>
                    <button type="submit">Send Information</button>
                </form>
            """    
            st.markdown(contact_form, unsafe_allow_html=True)

            def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            local_css("style/style.css")
#==========================================================================================================

# ONCHOCERCIASIS SECTION
# DATASET INCLUDES LOA LOA ENDEMICITY -- RISK OF SAEs

#==========================================================================================================
if "Onchocerciasis" in ntd_disease:

    # st.write("""African Programme of Onchocerciasis Control priority countries:
    #     Angola, Burundi, Cameroon, Central African Republic, Chad, Congo, Democratic
    #     Republic of the Congo, Equatorial Guinea, Ethiopia, Liberia, Malawi,
    #     Nigeria, South Sudan, Sudan, Tanzania, and Uganda. 
    #     Donation of ivermectin by Merck through the Mectizan Donation Program.    
    # """)    

    # Onchocerciasis technical inputs (Quality of life loss)
    itch_quality_loss = 0.068
    vision_quality_loss = 0.282
    blindness_quality_loss = 0.594
    blindness_mortality_loss = 8 # 8 years of life lost due to blindness
    ov1 = ["productivity loss patient", .38, .79, .19 ]
    ov2 = ['productivity loss carer', .05, .10, 0]
    ov3 = ["daly weights (untreated)", .282, .6, .068]
    ov4 = ["daly weights (treated)", .224, .488, .068]

    onchocerciasis_notes = """
    -   80% African country target for elimination 2025 (African Programme for Onchocerciasis Control)
    -   Integrated approach under Expanded Special Project for Elimination of NTDS (ESPEN)
    -   Severe skin itching is associated with 19% productivity loss
    -   Moderate skin itching is associated with 10% productivity loss
    -   Mild skin itching - no productivity loss
    -   Blindness patient - 79% productivity loss; Caregiver - 10% productivity loss
    -   Reduced vision patient - 38% productivity loss; Caregiver - 5% productivity loss
    -   Blindness premature mortality - 12 years [8-16]
    -   Visual impairment duration - 7.9 years
    -   Itching duration - 5.8 years
    -   Agricultural wages used to estimate productivity -- adjusted for equity
    -   Approach -- Human Capital
    -   Strategies -- Loa First test and treat (add costs for rapid test (Loascope))
    -   This captures the 3% of the population at risk of post-ivermectin SAEs
    -   Check which geographical zones will need this strategy from maps i.e. co-endemic Loa Loa and OV
    -   Get Loaiasis data from ESPEN and merge them with the OV dataset. Check for microfilaremia prevalence > 20%.
    -   If prevalence > 20%, assume 2-9% of the adult population have >30,000 mf/ml and are not eligible for ivermectin
    -   Assumption is that in these areas, TNT will be done on everyone leading to higher costs
    -   Loaisis positive will be treated with doxycycline [kills wolbachia]
    -   Routine MDA treatment period assumed to be 20 years
    -   Assume single administration per year (aCDTI)
    -   Can do simulations with bi-annual strategy (more data intensive) - will shorten time to provision operational threshold for treatment interruption followed by surveillance (pOTTIS) bCDTI reduces time by 35% in mesoendemic areas
    -   Select menu for endemicity status for OV [Mesoendemic, hyperendemic, highly-hyperendemic]
    -   Assumed therapeutic coverage 80%
    -   Assume 7% microfilarial production decline per ivermectin dose [Gardon, Turner]
    -   Assumed pOTTIS <1.4% [Turner]
    """
    with tabs[0]:

        st.markdown(translate_text("Please read this section before using the tool"))
        start = st.expander(translate_text("Click to read"))
        start.write(translate_text(about_tool))

        gen_notes = st.expander(translate_text("General notes for onchocerciasis module"))
        gen_notes.write(translate_markdown(onchocerciasis_notes))

        st.markdown("-----")

    with tabs[1]:
    
        # #========================================================================================
        # The country_espen dataframe contains LF status details
        country_espen = pd.read_csv("/Users/wobiero/Desktop/LF_Lymphasim/oncho.csv")
        country_espen = country_espen[country_espen["Year"]==2020].copy()
        country_espen = country_espen[country_espen["ADMIN0"]==country] 
        country_espen = country_espen.drop_duplicates(subset=["IU_CODE"])
        reduce_mem_usage(country_espen)
        non_endemic_ov = ['The Gambia', 'Madagascar', 'Sao Tome and Principe', 'Zimbabwe', 'Zambia', 'South Africa',
                        'Mauritania', 'Seychelles', 'Eritrea', 'Lesotho', 'Namibia', 'Comoros', 'Cabo Verde',
                        'Eswatini', 'Mauritius', 'Algeria', 'Rwanda', 'Kenya', 'Botswana']
    
        if country in non_endemic_ov:
            st.warning(translate_text(f"{country} is non-endemic for onchocerciasis. Please select another country!"))
            st.stop()
        
        see_data_1 = st.expander(translate_text('Click here to see country default data'))
        with see_data_1:
            show_df(country_inputs)
        st.write(translate_text(f'<p style="color:red;"> Please verify onchocerciasis prevalence rates with the {country} NTD programme. Some of the estimates maybe incorrect.'), unsafe_allow_html=True)   
        see_data_2 = st.expander(translate_text('Click here to see the raw 2020 ESPEN dataset'))
        with see_data_2:
            show_df(country_espen)
            
        def loa_endemicity(country):
            """
            Check the Loa loa status within a country and provide a summary message.
            
            Parameters
            ----------
            country : str
                The name of the country to check for Loa loa endemicity.
            
            Returns
            -------
            str
                A summary message indicating the Loa loa endemicity status within the country.
            """
            x = list(set(country_espen["Endemicity_loa"][country_espen["ADMIN0"]==country]))
            if any(k in x for k in ['Hypo-endemic', 'Meso-endemic', 'Hyper-endemic']):
                loa_ius = len(list(country_espen[country_espen["ADMIN0"]=="Angola"].query("Endemicity_loa in('Meso-endemic', 'Hypo-endemic', 'Hyper-endemic')")["IUs_NAME"]))
                loa_summary = translate_text(f"""
                Loa Loa is endemic in {loa_ius} implementing units in {country}. 
                This has programmatic implications for onchocerciasis MDA and length of MDA. 
                Please use the ESPEN dataset to identify the affected units.""")
                return loa_summary
            else:
                loa_summary = translate_text(f"Loa loa is non-endemic in {country}")
                return loa_summary

        loa_status = st.expander(translate_text("Loa loa endemicity status"))
        with loa_status:
            st.write(loa_endemicity(country))

        #==================================================================================
        st.markdown("-----")
        #===================================================================================

        @st.cache
        def unit_endemicity(admin_level, level):
            """
            Check the onchocerciasis endemicity status of a given administrative unit
            in a geographical area according to the ESPEN (2020) database. If the
            status is "Non-endemic", "Not reported", or "Unknown (consider Oncho
            Elimination Mapping)", issue a warning message and stop the execution.
            If the status is "Endemic (under post-intervention surveillance)",
            "Endemic (under MDA)", or a combination of both, issue an appropriate
            message.

            Parameters
            ----------
            admin_level : str
                The name of the administrative level (e.g., "IUs", "LCs", "LGAs",
                "States", etc.) used in the ESPEN database.
            level : str
                The name of the specific administrative unit for which to check the
                onchocerciasis endemicity status.

            Returns
            -------
            None
            """
            unit_status = set(country_espen["Endemicity"][country_espen[admin_level] == level])
            if len(unit_status) == 1:
                status = unit_status.pop()
                if status == "Non-endemic":
                    st.warning(translate_text(f"{level}, {country} is non-endemic for onchocerciasis according to ESPEN (2020). Please select another geographical area."))
                    st.stop()
                elif status == "Not reported":
                    st.warning(translate_text(f"{level}, {country} is classified as status not reported for onchocerciasis according to ESPEN (2020). Please select another geographical area."))
                    st.stop()
                elif status == 'Unknown (consider Oncho Elimination Mapping)':
                    st.warning(translate_text(f"{level}, {country} is classified as unknown for onchocerciasis according to ESPEN (2020). Please select another geographical area."))
                    st.stop()

            unknown_statuses = ["Unknown (under LF MDA)", "Not reported", 'Unknown (consider Oncho Elimination Mapping)']
            if any(status in unknown_statuses for status in unit_status):
                unknown_status = country_espen["IUs_NAME"][(country_espen[admin_level] == level) &
                                            (country_espen["Endemicity"].isin(unknown_statuses))]
                st.warning(translate_text(f"""
                    The onchocerciasis endemicity status in the following subunit(s): {', '.join(str(x) for x in unknown_status)} is unknown (under LF MDA), not reported,
                    or for consideration of Oncho elimination mapping.
                    Contact the {level} or {country} NTD programme to get more details about the implementing unit(s) before running any analyses. The default
                    in this tool is that these units are endemic since the ESPEN database shows they are targeted for MDA.
                    """))

            remaining_statuses = unit_status - set(unknown_statuses)
            if len(remaining_statuses) == 1:
                status = remaining_statuses.pop()
                if status == "Non-endemic":
                    st.warning(translate_text(f"{level}, {country} onchocerciasis programme has some delays according to ESPEN (2020)."))
                elif status == 'Endemic (under post-intervention surveillance)':
                    st.warning(translate_text(f"{level}, {country} is in the post-MDA surveillance phase according to ESPEN (2020)"))
                elif status == "Endemic (under MDA)":
                    st.write(translate_text(f"{level} is under MDA"))

        # Admin 1 dropdown menu
        st.sidebar.markdown(translate_text("#### Select geographical unit for analyses."))
        admin1 = sorted(list(set(country_espen["ADMIN1"])))
        admin1.insert(0, translate_text("National Level"))
        country_admin1 = st.sidebar.selectbox(translate_text("Administrative Unit 1"), admin1)
        program_summary = st.expander(translate_text("Country onchocerciasis status"))
        with st.expander(translate_text("Geographical Unit Summary")):
            if translate_text("National Level") in country_admin1:
                pop_req_mda = sum(country_espen["PopReq"])
                pop_tag_mda = sum(country_espen["PopTrg"])
                pop_trt_mda = sum(country_espen["PopTreat"])
                mf_baseline = sum(p * pop for p, pop in zip(country_espen["ov_prev_mean"], country_espen["PopReq"]))/(country_espen["PopReq"].sum())
                rounds_pre_tas = mda_rounds_left("ADMIN0", country, status = "Cum_MDA")
                sub_optimal_coverage = coverage_issues(country, lower=0, upper=80)
                
                if len(sub_optimal_coverage) == 0:
                    st.write(translate_text(f"""
                    The MDA target population at the national level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country} is {rounds_pre_tas}
                    according to ESPEN (2020). Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))
                else:
                    st.write(translate_text(f"""
                    The MDA target population at the national level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country} is {rounds_pre_tas}
                    according to ESPEN (2020). Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level.
                    The following implementing units had sub-optimal MDA geographical coverage (<80%): _{';  '.join(sub_optimal_coverage).title()}_."""))
            else:
                unit_endemicity("ADMIN1", country_admin1)
                admin2 = sorted(list(set(country_espen["ADMIN2"][country_espen["ADMIN1"]==country_admin1])))
                admin2.insert(0, translate_text("Sub-National Level 1"))
                country_admin2 = st.sidebar.selectbox(translate_text("Administrative Unit 2"), admin2)

                if translate_text("Sub-National Level 1") in country_admin2:
                    target = country_espen["PopReq"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopReq"]!=0)]
                    ov_prevalence = country_espen["ov_prev_mean"][country_espen["ADMIN1"]==country_admin1]
                    pop_req_mda = (sum(country_espen["PopReq"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopReq"]!=0)]))
                    pop_tag_mda = (sum(country_espen["PopTrg"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopTrg"]!=0)]))
                    pop_trt_mda = (sum(country_espen["PopTreat"][(country_espen["ADMIN1"]==country_admin1) &(country_espen["PopTreat"]!=0)]))
                    mf_baseline = sum(p * pop for p, pop in zip(ov_prevalence, target))/target.sum()
                    rounds_pre_tas = mda_rounds_left("ADMIN1", country_admin1, status="Cum_MDA")
                    
                    st.write(translate_text(f"""
                    The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                    The maximum number of MDA rounds still required in at least one implementing unit in {country_admin1.title()} is {rounds_pre_tas}
                    according to ESPEN. Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                    Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))

                if not translate_text("Sub-National Level 1") in country_admin2:
                    unit_endemicity("ADMIN2", country_admin2)
                    admin3 = sorted(list(set(country_espen["IUs_NAME"][country_espen["ADMIN2"]==country_admin2])))
                    admin3.insert(0, translate_text("Sub-National Level 2"))
                    country_iu = st.sidebar.selectbox(translate_text("Implementing Unit"), admin3)

                    if translate_text("Sub-National Level 2") in country_iu:
                        target = country_espen["PopReq"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopReq"]!=0)]
                        st.write(target)
                        ov_prevalence = country_espen["ov_prev_mean"][country_espen["ADMIN2"]==country_admin1]
            
                        pop_req_mda = (sum(country_espen["PopReq"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopReq"]!=0)]))
                        pop_tag_mda = (sum(country_espen["PopTrg"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopTrg"]!=0)]))
                        pop_trt_mda = (sum(country_espen["PopTreat"][(country_espen["ADMIN2"]==country_admin2) &(country_espen["PopTreat"]!=0)]))
                        
                        mf_baseline = sum(p * pop for p, pop in zip(ov_prevalence, target))/target.sum()
                        rounds_pre_tas = mda_rounds_left("ADMIN2", country_admin2, status="Cum_MDA")
                        
                        st.write(translate_text(f"""
                        The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                        The maximum number of MDA rounds still required in at least one implementing unit in {country_admin2.title()} is {rounds_pre_tas}
                        according to ESPEN. Please use the drop-down menus, tables and interactive maps to identify this/these unit(s).
                        Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))

                    if not translate_text("Sub-National Level 2") in country_iu:   
                        progress_status_iu = list(country_espen["Endemicity"][country_espen["IUs_NAME"]==country_iu])
                        unit_endemicity("IUs_NAME", country_iu)
                        
                        pop_req_mda = list(country_espen["PopReq"][(country_espen["IUs_NAME"]==country_iu)])[0] #Population requiring MDA
                        pop_tag_mda = list(country_espen["PopTrg"][(country_espen["IUs_NAME"]==country_iu)])[0] #Population targeted with MDA
                        pop_trt_mda = list(country_espen["PopTreat"][(country_espen["IUs_NAME"]==country_iu)])[0] #Population that received MDA
                        rounds_pre_tas = mda_rounds_left("IUs_NAME", country_iu, status="Cum_MDA")
                        mf_baseline = list(country_espen["ov_prev_mean"][(country_espen["IUs_NAME"]==country_iu)])[0]
                        
                        st.markdown(translate_text(f"""
                        The MDA target population at this administrative level is {pop_req_mda:,.0f}. 
                        The maximum number of MDA rounds still required in {country_iu.title()} is {rounds_pre_tas}
                        according to ESPEN. 
                        Leave the dropdown menu unchanged if you want to conduct analyses at this administrative level."""))

                        if 'Non-endemic' in progress_status_iu:
                            st.warning(translate_text(f"{country_iu} is non_endemic for onchocerciasis. Please select another IU."))
        
        comp_class = st.sidebar.expander(translate_text("Onchocerciasis Program Details"))
        ov_rounds = country_espen["Cum_MDA"].max()
        
        with comp_class:
            country_status = st.selectbox(
                translate_text("Select Oncho Program Status"),
                ('Non-endemic', 'Not reported', 'Endemic (MDA not delivered)', 'Endemic (under MDA)', 
                'Unknown (consider Oncho Elimination Mapping)', 'Endemic (under post-intervention surveillance)', 
                'Unknown (under LF MDA)'),
                index = 1
            )
            # The list below shows the decline in OV microfilarial prevalence with MDA assuming 80% coverage
            # These values are based on EpiOncho simulations
            mf_prev_decline = [79.93380406, 42.46690203, 40.61341571, 38.62753751, 35.84730803, 32.93468667, 30.94880847, 
                                28.69814651, 26.44748455, 24.59399823, 23.00529568, 21.68137688, 20.2250662, 18.9011474, 
                                17.70962048, 16.78287732, 16.18558951, 14.38985418, 12.46142868, 11.13018256, 9.931479339, 
                                8.732923388, 7.534293801, 6.46835438, 5.999373434, 5.265159426, 4.796546655, 3.80455408, 
                                3.406072106, 2.874762808, 2.609108159, 2.210626186, 1.944971537, 1.612903226, 1.129032258, 
                                0.806451613, 0.64516129, 0.403225806, 0.080645161, 0.080645161]
            time_to_elimination = [*range(1, len(mf_prev_decline), 1)]
            time_to_elimination_2 = time_to_elimination.copy()
            time_to_elimination.reverse()

            @st.cache
            def nearest_index(arr, value):
                """
                This function estimates the remaining time based on starting mf prevalence
                """
                differences = np.abs(np.array(arr) - value)
                index = np.argmin(differences)
                return index
            surv = 6 # This is set at six years post-TAS according to the literature
            normal_prog_length = 20 # Normal length of Onchocerciasis MDA rounds
            time_remaining = max(0, (normal_prog_length - ov_rounds) + surv) # Conservative estimate
            
            st.write(translate_text(f"The at risk population is: {pop_req_mda:,}"))
            st.write(translate_text(f"The baseline microfilaria prevalence at this level is: {mf_baseline * 100:,.2f}%"))
            _,slider_col,_ = st.columns([.02, .96, .02])
            
            with slider_col:
                # Microfilaria prevalence baseline set to zero if countries have validated elimination
                
                mf_baseline = round(mf_baseline * 100)
                if country_status == 'Non-endemic' or country_status == 'Endemic (under post-intervention surveillance)':
                    mf_baseline = 0
                if "mf_prevalence" not in st.session_state:
                    st.session_state["mf_prevalence"] = mf_baseline
                
                mf_prevalence = st.session_state["mf_prevalence"] = st.slider(translate_text("% Current microfilaria prevalence"),0,100, mf_baseline)
                mf_baseline = mf_prevalence
            # Use nearest_index function and the reversed time list to get years left to elimination assuming 80% MDA coverage
            elim_years = time_to_elimination[nearest_index(mf_prev_decline, mf_baseline)] 
            
            default_ov_coverage = int(country_espen["Cov"][country_espen["ADMIN0"]==country].mean())
            ov_mda_coverage = st.session_state["default_ov_coverage"] = st.slider(translate_text("% MDA coverage"),
                                                                                min_value = 0,
                                                                                max_value = 100,
                                                                                value= default_ov_coverage)
            if "time_remaining" not in st.session_state:
                st.session_state.time_remaining = 0
            if "elim_years" not in st.session_state:
                st.session_state.elim_years = 0

            if ov_mda_coverage < 30:
                st.session_state.time_remaining += 5
                st.session_state.elim_years += 5
            elif ov_mda_coverage < 80:
                st.session_state.time_remaining += 3
                st.session_state.elim_years += 3
            
            #ov_mda_frequency = st.radio(translate_text("MDA Frequency"), (translate_text("Yearly"), translate_text("Bi-annual"), translate_text("Quarterly")), index=0)
            ov_mda_frequency = st.radio(translate_text("MDA Frequency"), ("Yearly", "Bi-annual", "Quarterly"))
            if ov_mda_frequency == "Bi-annual" or ov_mda_frequency == "Quarterly":
                st.session_state.time_remaining /= 1.54271362
                st.session_state.elim_years /= 1.54271362
                st.session_state.elim_years = int(round(elim_years))
            
            ov_endemic_status = st.radio(translate_text("Endemicity"), (translate_text("Hyperendemic"), translate_text("Mesoendemic"), translate_text("Hypoendemic")))
            if ov_endemic_status == "Hyperendemic":
                st.session_state.time_remaining += 5
            elif ov_endemic_status == "Mesoendemic":
                st.session_state.time_remaining +=3

            ov_bio_climatic = st.radio(translate_text("Bio-climatic zone"), (translate_text("Savannah"), translate_text("Forest")))
            
            #If forest, divide blindness by four
            vector_control = st.radio(translate_text("Vector control"), (translate_text("No"), translate_text("Yes")))
            if vector_control == "Yes":
                st.session_state.time_remaining *= .9 # We assume that vector control is only 10% effective. Can be adjusted with better data
                st.session_state.elim_years *= .9
                st.session_state.elim_years = int(round(elim_years))
            contiguous_endemicity = st.radio(translate_text("Cross-border endemicity"), (translate_text("No"), translate_text("Yes")))
            if contiguous_endemicity == "Yes":
                st.session_state.time_remaining += 2
                st.session_state.elim_years += 2
                
            min_elim_time = int(round(elim_years + 2020))   
            range_selection = st.slider(translate_text("Select time horizon in years"), 2020, 2100, min_elim_time)
            time_horizon = range_selection - 2020 # Time horizon to be used for discounting and totals
            ov_surveillance = st.slider(translate_text("Select surveillance period in years"), 0,20, surv)

        st.warning(translate_text(f"""The minimum time to elimination based on the defaults/adjustments done in the side bar is {round(elim_years)} years.
                Elimination is unlikely to happen with MDA coverage less than 80%."""))
#=============================================================================================================================
        country_expander = st.sidebar.expander(translate_text("Insert country parameters"))

        with country_expander:
            econ_values()
            hh_costs_adl = st.number_input("Insert household costs ADL", value=10)
            hh_costs_lym = st.number_input("Insert household costs lymphedema", value=10)
            hh_costs_hyd = st.number_input("Insert household costs hydrocele", value=10)

        pop_expander = st.sidebar.expander(translate_text("Please adjust default population values"))
        # Population pyramid (%)for SSA in 5-year gaps. Over 70 combined
        ssa_pop_pyramid = [15.6, 14.0, 12.4,10.7, 9.2, 7.8,
                            6.6, 5.6, 4.5, 3.6, 2.8, 2.3, 1.7, 1.3, 1.9]
        
        with pop_expander:
            pop_distribution()
            
        cost_expander = st.sidebar.expander(translate_text("Insert program costs"))

        with cost_expander:
            # Remember to allocate the costs appropriately if shared
            personnel_1 = st.number_input("Project personnel costs", value = 1.0)
            personnel_2 = st.number_input("MOH personnel costs", value = 1.0)
            personnel_3 = st.number_input("Volunteer personnel costs", value = 1.0)

            capital_1 = st.number_input("Building (annualized) costs", value = 1.0)

            consumables_1 = st.number_input("MDA drug costs", value = 1.0)
            consumables_2 = st.number_input("Consumables costs", value = 1.0)

            training_1 = st.number_input("MDA training costs", value = 1.0)
            training_2 = st.number_input("TAS training costs", value = 1.0)

            transport_1 = st.number_input("Vehicle purchase (annualized) costs", value = 1.0) #Remember to adjust by project share
            transport_2 = st.number_input("Vehicle leasing costs", value = 1.0)

            meetings_1 = st.number_input("Planning meeting costs", value = 1.0)
            meetings_2 = st.number_input("Mapping meeting costs", value = 1.0)
            meetings_3 = st.number_input("TAS meeting costs", value = 1.0)
            meetings_4 = st.number_input("Other meetings incl. conferences", value = 1.0)

            mda_cost = st.number_input("MDA dose costs", value=1.15)
            tas_cost = st.number_input("Transmission assessment survey costs", value=1)
            post_mda_cost = st.number_input("Post MDA surveillance costs", value=1)

            others_1 = st.number_input("Others e.g. per diem costs", value=1)
            total_prog_cost = (personnel_1 + personnel_2 + personnel_3 + capital_1 + consumables_1 +
                consumables_2 + training_1 + training_2 + transport_1 + transport_2 +
                meetings_1 + meetings_2 + meetings_3 + meetings_4 + mda_cost +
                tas_cost + post_mda_cost + others_1)

        prog_cost_summary = {
            "Project personnel costs": personnel_1 ,
            "MOH personnel costs": personnel_2 ,
            "Volunteer personnel costs": personnel_3,
            "Building (annualized) costs":capital_1,
            "MDA drug costs": consumables_1,
            "Consumables costs": consumables_2,
            "MDA training costs": training_1,
            "TAS training costs": training_2,
            "Vehicle purchase (annualized) costs": transport_1,
            "Vehicle leasing costs": transport_2,
            "Planning meeting costs": meetings_1,
            "Mapping meeting costs": meetings_2,
            "TAS meeting costs": meetings_3,
            "Other meetings incl. conferences": meetings_4,
            "Mass drug administration costs": mda_cost,
            "Transmission assessment survey costs": tas_cost, 
            "Post MDA surveillance costs": post_mda_cost,
            "Others e.g. entomological and serological surveys": others_1,
            "Total": total_prog_cost
        }

        prog_cost_dataframe = pd.DataFrame(prog_cost_summary.items(), columns=["Category", "Estimate"])
        prog_cost_expander = st.expander(translate_text("Summary of Progammatic Costs"))
        prog_cost_expander.write(prog_cost_dataframe)
        
        # Extract mean OPD costs based on WHO CHOICE to be used for cost calculations

        mean_opd_visit_cost = country_inputs.at[country_inputs[country_inputs["Country"]==country].index[0], "Health Centre-no beds"]
        
        patient_costs = st.sidebar.expander(translate_text("Average patient costs"))

        with patient_costs:
            mean_visit_cost = st.number_input("Mean patient visit cost ($)", value = mean_opd_visit_cost)
            medication_costs = st.number_input("Drug costs for OPD visit", value = 1)
            laboratory_costs = st.number_input("Laboratory costs for OPD visit", value=1)
            specialist_costs = st.number_input("Ophthalmology p.a.", value=1) 
            skincare_costs = st.number_input("Dermatology costs p.a.", value=1)
            consultation_costs = st.number_input("Specialist consultation costs", value=1)
            travel_costs = st.number_input("Avg. travel costs", value=1)
            avg_time_at_clinic = st.number_input("Average time spent at facility in hours", value=1)
            time_costs_routine = avg_time_at_clinic * daily_wage()[1]

            patient_visit_costs = (
                medication_costs + laboratory_costs + consultation_costs + travel_costs +
                specialist_costs + skincare_costs + time_costs_routine
            )
            medical_cost_inflation = (3 + country_inputs['Inflation rate (consumer prices) (%)'][country_inputs["Country"]==country])/100

        technical_expander = st.sidebar.expander(translate_text("Technical parameters (In-built defaults))"))
        with technical_expander:

            # Disability weights from the GBD study
            dw_lymphedema = st.number_input("Disability weight lymphedema(default=.105)", value=.105)
            dw_hydrocele = st.number_input("Disability weight hydrocele(default=.073)", value=.073)
            avg_onset = st.number_input("Age of symptom onset LF(default=20)", value=20)
            #Economic defaults
            disc_costs = st.number_input("Discount rate costs", value=0.04)
            disc_effects = st.number_input("Discount rate effects", value=0.04)

            # Health system utilization
            hyd_visits = st.number_input("Hydrocele clinic visits p.a.", value=2)
            lym_visits = st.number_input("Lymphedema clinic visits p.a.", value=4)
            adl_visits = st.number_input("ADL clinic visits p.a.", value=4.65)
            adl_episode = st.number_input("Duration of ADL episode", value=3.93)

            # Productivity loses
            prod_loss_hyd = st.number_input("Productivity loss hydrocele", value=.15)
            prod_loss_lym = st.number_input("Productivity loss lymphedema", value=.18)
            prod_loss_adl = st.number_input("Productivity loss ADL", value=.78)

            # Other parameters:
            at_risk_pop = st.number_input("At risk population(default=10%)", value=pop_req_mda)
            lf_to_lymph = st.number_input("% of LF infections->lymphedema(default=12.5%)", value=.125)
            lf_to_hydro = st.number_input("% of LF infections->hydrocele(default=20.8%)", value=.208)
            lf_to_subcl = st.number_input("% of LF infections subclinical(default=66.7%)", value=.667)
            daly_av_hydr = st.number_input("DALY Averted per prevented hydrocele(default=2.3)", value=2.3)
            daly_av_lymh = st.number_input("DALY averted per prevented lymphedema(default=3.3)", value=3.3)
#=============================================================================================================================
        country_maps = st.expander(translate_text(f"Click here to see and download the {country} MDA programme status map"))
        with country_maps:

            try:
                country_map_1 = Image.open("/Users/wobiero/Desktop/LF_Lymphasim/static_oncho_maps/"+country+".png")
                st.image(country_map_1)
                st.download_button(label = f"Download {country} map",
                                    data = open("/Users/wobiero/Desktop/LF_Lymphasim/static_oncho_maps/"+country+".png", 'rb').read(),
                                    file_name =country+".png",
                                    mime = "image/png")
            except IOError:
                st.write(translate_text(f"Static map for {country} not found. Please check source directory and rerun tool."))

        interactive_map = st.expander(translate_text(f"Click here if you want to see the interactive map for {country}"))
        with interactive_map:
            try:
                html_map = open("/Users/wobiero/Desktop/LF_Lymphasim/oncho_maps/"+country+".html", "r", encoding='utf-8')
                source_code =html_map.read()
                components.html(source_code, width=1000, height=1500)
            except IOError:
                st.write(translate_text(f"HTML map for {country} not found. Please check source directory and rerun."))
            st.write("")
        #=============================================================================================
    with tabs[2]:
        # Antimacrofilarial impact of drugs
        mesoendemic = ["Mesoendemic" , 17, 11, 11] #40%
        hyperendemic = ["Hyperendemic" ,25, 16, 11] #60%
        highly_hyperendemic = ["Highly-hyperendemic",50, 26, 26] #80%
        ov_columns = ["Endemicity","aCDTI", "bCDTI", "aCDTM"]
        ov_endemicity = pd.DataFrame([mesoendemic, hyperendemic, highly_hyperendemic],
        columns = ov_columns)
        #st.dataframe(ov_endemicity.style.highlight_max(axis=0))
        # Epilepsy [Vinkeles Melchers et al](https://idpjournal.biomedcentral.com/articles/10.1186/s40249-018-0481-9)
        # Internal check -- total estimated OAE 381,000 (check this with data)
        # Estimate years lived with disability -- according to this paper OAE accounts for 13% of total Oncho YLD
        # 0.4% increase in epilepsy prevalence for every 10% increase in oncho prevalence
        # Assume 1.23% OAE prevalence in West Africa -- get target pop and multiply by this
        # Oncho Phase 1: 12- 15 years; Phase 2 (Post-Rx surveillance): 3-5 years; Phase 3 (post-elimination surveillance); 3-5 years
        # Minimum 18 years
        # Peak age itching - 20years then declines
        #Hyperendemic >45% nodules; mesoendemic 20-45% nodules; hypoendemic <20% nodules [Coffeng et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3623701/)
        
        oncho_cases = pop_req_mda * .04839 # Ratio 20938100/432715463
        
        @dataclass
        class OnchoInputs:
            n_iterations = 1000

            mean_age: float = 29.7
            mean_age_std: float = mean_age * .3

            oncho_cases: float = oncho_cases
            oncho_cases_std: float = oncho_cases * .297

            oncho_vision: float = oncho_cases * .055
            oncho_vision_std: float = oncho_vision * .194

            oncho_vision_visits: float = oncho_vision * 2.0
            oncho_vision_visits_std: float = oncho_vision_visits * .5

            oncho_blind: float = oncho_cases * .019
            oncho_blind_std: float = oncho_blind * .34

            oncho_skin: float = oncho_cases * .70
            oncho_skin_std: float = oncho_skin * .16

            severe_itch_cases: float = oncho_skin * .31
            severe_itch_cases_std: float = severe_itch_cases * .1

            severe_skin_visits: float = severe_itch_cases * 4.0 # Assumes treatment every three months
            severe_skin_visits_std: float = severe_skin_visits * .3

            other_skin: float = oncho_skin * .69
            other_skin_std: float = other_skin * .1

            oae_cases: float = oncho_cases *.0138 #oae - ochocerciasis-associated epilepsy
            oae_cases_std: float = oae_cases * .33

            oae_visits_pa: float = 3 # Consider changing this to lognormal
            oae_visits_pa_std: float = oae_visits_pa * .1

            opd_visits_pa: float = 0.4 # Consider changing this to lognormal
            opd_visits_pa_std: float = 0.25

            consultation_time: float = 0.0625 # Average consultation time assumed to be 30 minutes in an 8-hour working day
            consultation_time_std: float = 0.0125 # Consider changing this to lognormal

            epilepsy_rate: float = .0036 # Background rate of total epilepsy cases in Africa
            epilepsy_rate_std: float = .00054

            decline_oae_rx_pa: float = .035 # Decline of OAE with ivermectin p.a. -- only applies to areas with MDA
            decline_oae_rx_pa_std: float = decline_oae_rx_pa * .5

            dw_other_itch: float = .038 # Assume same as eczema
            dw_other_itch_std: float = dw_other_itch * .2

            dw_severe_itch: float = .187
            dw_severe_itch_std: float = dw_severe_itch * .1

            dw_blind:float = .60 # This dropped from .60 to .195 in the 2020 GBD but has been criticised. 
            dw_blind_std:float =dw_blind * .1

            dw_epilepsy: float = 0.336
            dw_epilepsy_std: float = dw_epilepsy * .1

            dw_low_vision: float = .033
            dw_low_vision_std: float = dw_low_vision * .1

            prod_loss_epilepsy:float = .36
            prod_loss_epilepsy_std: float = prod_loss_epilepsy * .2
            
            prod_loss_severe_itch: float = .19
            prod_loss_severe_itch_std: float = prod_loss_severe_itch * .1

            prod_loss_blind: float = .79
            prod_loss_blind_std: float = prod_loss_blind * .1

            prod_loss_low_vision: float = .38
            prod_loss_low_vision_std: float = prod_loss_low_vision * .1

            prod_loss_caretaker_lv: float = .05
            prod_loss_caretaker_lv_std: float = prod_loss_caretaker_lv * .1

            prod_loss_caretaker_blind: float = .10
            prod_loss_caretaker_blind_std: float = prod_loss_caretaker_blind * .1

            morbidity_reduction_mda: float = .9
            morbidity_reduction_mda_std: float = .05
            
            reduced_le: float = 4.0 # Reduced life expectancy due to onchocerciasis -- hyperendemic only
            reduced_le_std: float = 1.5
            
        oncho_data = OnchoInputs()
        from dataclasses import asdict
        # Convert into dictionary and transpose -- to use this under the technical tab
        data_dict = asdict(oncho_data)
        data_dict = {k: [v] for k,v in data_dict.items()}
        oncho_data_technical = pd.DataFrame(data_dict, index=[0]).T
        oncho_data_technical = oncho_data_technical.rename(columns={0: 'Value'}).reset_index().rename(columns={'index': 'Indicator'})

        
        def oncho_simulation_inputs(oncho_data):
            mean_age = random_normal_positive(oncho_data.mean_age, oncho_data.mean_age_std)
            oncho_cases = random_normal_positive(oncho_data.oncho_cases, oncho_data.oncho_cases_std)
            
            oncho_vision = random_normal_positive(oncho_data.oncho_vision, oncho_data.oncho_vision_std)
            oncho_blind = random_normal_positive(oncho_data.oncho_blind, oncho_data.oncho_blind_std)
            oncho_vision_visits = random_normal_positive(oncho_data.oncho_vision_visits, oncho_data.oncho_vision_visits_std)

            oncho_skin = random_normal_positive(oncho_data.oncho_skin, oncho_data.oncho_skin_std)
            severe_itch_cases = random_normal_positive(oncho_data.severe_itch_cases, oncho_data.severe_itch_cases_std)
            other_skin = random_normal_positive(oncho_data.other_skin, oncho_data.other_skin_std)
            severe_skin_visits = random_normal_positive(oncho_data.severe_skin_visits, oncho_data.severe_skin_visits_std)

            oae_cases = random_normal_positive(oncho_data.oae_cases, oncho_data.oae_cases_std)
            oae_visits_pa = random_normal_positive(oncho_data.oae_visits_pa, oncho_data.oae_visits_pa_std)
            
            epilepsy_rate = gamma_positive(oncho_data.epilepsy_rate, oncho_data.epilepsy_rate_std)
            decline_oae_rx_pa = gamma_positive(oncho_data.decline_oae_rx_pa, oncho_data.decline_oae_rx_pa_std)
            
            opd_visits_pa = random_normal_positive(oncho_data.opd_visits_pa, oncho_data.opd_visits_pa_std)
            consultation_time = random_normal_positive(oncho_data.consultation_time, oncho_data.consultation_time_std)
            
            # Disability weights for different conditions
            dw_epilepsy = random_normal_positive(oncho_data.dw_epilepsy, oncho_data.dw_epilepsy_std)
            dw_low_vision = random_normal_positive(oncho_data.dw_low_vision, oncho_data.dw_low_vision_std)
            dw_blind = random_normal_positive(oncho_data.dw_blind, oncho_data.dw_blind_std)
            dw_severe_itch = random_normal_positive(oncho_data.dw_severe_itch, oncho_data.dw_severe_itch_std)
            dw_other_itch = random_normal_positive(oncho_data.dw_other_itch, oncho_data.dw_other_itch_std)
            
            # Productivity losses
            prod_loss_severe_itch = random_normal_positive(oncho_data.prod_loss_severe_itch, oncho_data.prod_loss_severe_itch_std)
            prod_loss_blind = random_normal_positive(oncho_data.prod_loss_blind, oncho_data.prod_loss_blind_std)
            prod_loss_low_vision = random_normal_positive(oncho_data.prod_loss_low_vision, oncho_data.prod_loss_low_vision_std)
            prod_loss_caretaker_lv = random_normal_positive(oncho_data.prod_loss_caretaker_lv, oncho_data.prod_loss_caretaker_lv_std)
            prod_loss_caretaker_blind = random_normal_positive(oncho_data.prod_loss_caretaker_blind, oncho_data.prod_loss_caretaker_blind_std)
            prod_loss_epilepsy = random_normal_positive(oncho_data.prod_loss_epilepsy, oncho_data.prod_loss_epilepsy_std)
            
            # Others
            morbidity_reduction_mda = random_normal_positive(oncho_data.morbidity_reduction_mda, oncho_data.morbidity_reduction_mda_std)
            reduced_le = random_normal_positive(oncho_data.reduced_le, oncho_data.reduced_le_std)
            
            return (
                mean_age, oncho_cases, oncho_vision, oncho_blind, oncho_vision_visits,
                oncho_skin, severe_itch_cases, severe_skin_visits,
                other_skin, oae_cases, oae_visits_pa, epilepsy_rate,
                decline_oae_rx_pa, opd_visits_pa, consultation_time,
                dw_epilepsy, dw_low_vision, dw_blind, dw_severe_itch, dw_other_itch,
                prod_loss_severe_itch, prod_loss_blind, prod_loss_low_vision, prod_loss_caretaker_lv,
                prod_loss_caretaker_blind, prod_loss_epilepsy, morbidity_reduction_mda, reduced_le
            )
        
        
        def model_single_run(oncho_data):
            (
                mean_age, oncho_cases, oncho_vision, oncho_blind, oncho_vision_visits,
                oncho_skin, severe_itch_cases, severe_skin_visits,
                other_skin, oae_cases, oae_visits_pa, epilepsy_rate, 
                decline_oae_rx_pa, opd_visits_pa, consultation_time,
                dw_epilepsy, dw_low_vision, dw_blind, dw_severe_itch, dw_other_itch,
                prod_loss_severe_itch, prod_loss_blind, prod_loss_low_vision, prod_loss_caretaker_lv,
                prod_loss_caretaker_blind, prod_loss_epilepsy, morbidity_reduction_mda, reduced_le
            ) = oncho_simulation_inputs(oncho_data)

            return (
                mean_age, oncho_cases, oncho_vision, oncho_blind, oncho_vision_visits,
                oncho_skin, severe_itch_cases,
                severe_skin_visits, other_skin, oae_cases, oae_visits_pa, epilepsy_rate,
                decline_oae_rx_pa, opd_visits_pa, consultation_time,
                dw_epilepsy, dw_low_vision, dw_blind, dw_severe_itch, dw_other_itch,
                prod_loss_severe_itch, prod_loss_blind, prod_loss_low_vision, prod_loss_caretaker_lv,
                prod_loss_caretaker_blind, prod_loss_epilepsy, morbidity_reduction_mda, reduced_le
            ) 
        
        def monte_carlo_data(oncho_data):
            values = [model_single_run(oncho_data) for i in range(oncho_data.n_iterations)]
            df = pd.DataFrame(
                            values,
                            columns = [ 
                                        'mean_age','oncho_cases', 'oncho_vision', 'oncho_blind', 'oncho_vision_visits', 'oncho_skin',
                                        'severe_itch_cases', 'severe_skin_visits','other_skin', 'oae_cases', 'oae_visits_pa',
                                        'epilepsy_rate', 'decline_oae_rx_pa', 'opd_visits_pa', 'consultation_time',
                                        'dw_epilepsy', 'dw_low_vision', 'dw_blind', 'dw_severe_itch',
                                        'dw_other_itch', 'prod_loss_severe_itch', 'prod_loss_blind', 'prod_loss_low_vision',
                                        'prod_loss_caretaker_lv', 'prod_loss_caretaker_blind', 'prod_loss_epilepsy',
                                        'morbidity_reduction_mda', 'reduced_le'
                                ]
                                )
            return df
        simulated_df = monte_carlo_data(oncho_data)
        avg_le = country_inputs["Life_Expectancy"][country_inputs["Country"]==country].values[0]
        simulated_df["life_left"] = avg_le
        simulated_df["life_left"] = simulated_df["life_left"] - simulated_df["mean_age"] + simulated_df["reduced_le"]

        @st.cache
        def annual_vsl(life_left, gdp_target=1000, gdp_us= 63_530.6 ,vsl_usa = 9_600_000, elasticity = 1.0, disc=disc_costs):
            """
            Sum geometric series of rates
            Value of statistical life calculation
            Obtain gdp_target from country_input file
            Uses formula proposed by Viscusi et al [https://www.cambridge.org/core/journals/journal-of-benefit-cost-analysis/article/income-elasticities-and-global-values-of-a-statistical-life/5AE299883F668DCC265C41A377E1E063#s5]
            Elasticity of 1.0 makes sense. Can modify function if country elasticities are available
            Then estimate annual VSL
            """
            geom_rate_sum = sum([(1+disc)**(-i) for i in range(1, int(life_left+1))])
            vsl = vsl_usa * (gdp_target/gdp_us)**elasticity
            annual_vsl = vsl/geom_rate_sum
            return annual_vsl
        
        country_gdp = country_inputs["imf_estimate"][country_inputs["Country"]==country].values[0]
        
        simulated_df["annual_vsl"] = simulated_df.apply(lambda x: annual_vsl(x["life_left"],gdp_target=country_gdp), axis=1)
        simulated_df["yld_epilepsy"] = simulated_df["life_left"] * simulated_df["oae_cases"] * simulated_df["dw_epilepsy"]
        simulated_df["yld_severe_itch"] = simulated_df["life_left"] * simulated_df["severe_itch_cases"] * simulated_df["dw_severe_itch"]
        simulated_df["yld_other_skin"] = simulated_df["life_left"] * simulated_df["other_skin"] * simulated_df["dw_other_itch"]
        simulated_df["yld_low_vision"] = simulated_df["life_left"] * simulated_df["oncho_vision"] * simulated_df["dw_low_vision"]
        simulated_df["yld_blind"] = simulated_df["life_left"] * simulated_df["oncho_blind"] * simulated_df["dw_blind"]
        #Caretaker personnel years p.a.
        simulated_df["caretaker_time"] = (simulated_df["prod_loss_caretaker_lv"] * simulated_df["oncho_vision"]) + (simulated_df["prod_loss_caretaker_blind"] * simulated_df["oncho_blind"])
        #Patient personnel years p.a.
        simulated_df["patient_time"] = ((simulated_df["prod_loss_low_vision"] * simulated_df["oncho_vision"]) + 
                                        (simulated_df["prod_loss_blind"] * simulated_df["oncho_blind"]) +
                                        (simulated_df["prod_loss_severe_itch"] * simulated_df["severe_itch_cases"]) +
                                        (simulated_df["prod_loss_epilepsy"] * simulated_df["oae_cases"]))
        detailed_results = st.expander(translate_text("Monte Carlo Simulations Table. Click to see and download."))
        with detailed_results:
            st.dataframe(simulated_df.style.format("{:,.3f}"))
            st.download_button(translate_text("Download Summary"), simulated_df.to_csv(), mime="text/csv")
#=========================================================================================================================================
    # This section deals with summary results from the previous analyses.
#========================================================================================================================================
    with tabs[3]:   
        
        with st.expander(translate_text("Onchocerciasis mf prevalence reduction with MDA over time")):
            
            def ov_elim_potential(x, a, b):
                """
                Computes the potential for onchocerciasis elimination as a function of age.

                Parameters:
                x (array-like): An array of age values.
                a (float): The maximum potential value.
                b (float): The rate of decline in potential with age.
                noise (float): The standard deviation of the random noise to add to the potential values.

                Returns:
                An array of potential values.
                """
                return a * np.exp(-b * x)
            
            x = time_to_elimination_2[-elim_years:] 
            x.extend([x[-1] + i for i in range(1,ov_surveillance)])
            min_x = min(x)
            x = np.array([xi - min_x for xi in x])              
            y = mf_prev_decline[-elim_years:] 
            y.extend([y[-1] * 1/i for i in range(1,ov_surveillance)])
            y = np.array(y)
            
            params, params_covariance = curve_fit(ov_elim_potential, x, y)
            a_mean, b_mean = params
            a_stderr, b_stderr = np.sqrt(np.diag(params_covariance))
            a_rand = np.random.normal(a_mean, a_stderr, 100)
            b_rand = np.random.normal(b_mean, b_stderr, 100)
            
            master_list = []
            
            fig, ax = plt.subplots(figsize=(5.0,5.0))
            for a, b in zip(a_rand, b_rand):
                y_sim = ov_elim_potential(x, a, b)
                master_list.append(y_sim)
                plt.plot(x, y_sim, '-', alpha=0.3)

            plt.title(translate_text("Simulated Onchocerciasis Microfilarial Prevalence with MDA"), fontsize=9)
            plt.ylabel(translate_text("Microfilarial prevalence (%)"), fontsize=8)
            plt.xlabel(translate_text("Time to elimination"), fontsize=8)
            plt.grid(axis="both", alpha=.2)
            plt.axhline(y = 0.1, color="r", linestyle="--",label="Elimination target .1%")
            plt.axhline(y = 1.0, color="b", linestyle="--", lw=.5, alpha=.3)
            plt.axhline(y = 2.0, color="b", linestyle="--", lw=.5, alpha=.3)
            if min(y) < .01:
                plt.axvspan(x[- ov_surveillance], x[-1], facecolor="green", alpha=.3, label = "Surveillance period")
            plt.legend(loc="best", fancybox=True, fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            xtick_labels = ax.get_xticks() + 2020
        
            ax.set_xticklabels(xtick_labels.astype(int))

            st.pyplot(fig)
            st.write(translate_text("""<em>We assume an elimination threshold of .1% seroprevalence. 
                        Current studies suggest elimination could occur at a threshold of <2.0%, so this threshold might be restrictive.</em>"""), unsafe_allow_html=True)
            download_chart(fig)

            end_status = [x[-1] for x in master_list]
            ov_recrudescence_risk = len([x for x in end_status if x > 1.0])/len(end_status)
            ov_recrudescence_upper = len([x for x in end_status if x > 0.8])/len(end_status)
            ov_recrudescence_lower = len([x for x in end_status if x > 1.2])/len(end_status)

            if ov_mda_coverage < 80:
                st.markdown(translate_text(f'''<p style="background-color:{"#FFCCCB"};border-radius:4%; padding: 15px 15px;">{f"""According to mathematical models, OV 
                        elimination in the geographical unit is unlikely with a MDA coverage of {ov_mda_coverage}% and a time horizon
                            of {time_horizon} years. We assume here that elimination efforts will be enhanced."""}</p>'''), unsafe_allow_html=True)

            elif ov_recrudescence_risk < .9:
                st.write(translate_text(f"""The risk of OV recrudescence given the current programme status, a starting microfilaria prevalence of {mf_prevalence}%, a
                    program time horizon of {time_horizon} years, and a surveillance period of {ov_surveillance} years is 
                    {ov_recrudescence_risk * 100:.{1}f}%[{ov_recrudescence_lower * 100:.{1}f}%, {ov_recrudescence_upper * 100:.{1}f}%]."""))
            else:
                st.write(translate_text(f"""It is highly unlikely that OV will be eliminated in this location given a starting microfilaria prevalence of {mf_prevalence}%,
                    a MDA coverage of {ov_mda_coverage}% and a time horizon of {time_horizon} years."""))
        
        
        def total_annual_visits_1():
            """
            Calculates the total annual number of visits required for different health conditions in a given population.

            This function uses a set of simulated data stored in the `simulated_df` DataFrame to estimate the total number of annual visits required for the following health conditions: non-specific cases, severe skin disease, other skin diseases, poor vision, and onchocerciasis-associated eye disease (OAE).

            Returns
            -------
            tuple
                A tuple with six elements:
                - total_visits: the total annual number of visits required for all health conditions
                - non_specific: the annual number of visits required for non-specific cases
                - severe_skin: the annual number of visits required for severe skin disease
                - other_skin: the annual number of visits required for other skin diseases
                - poor_vision: the annual number of visits required for poor vision
                - oae_cases: the annual number of visits required for OAE cases

            Notes
            -----
            The `simulated_df` DataFrame should contain columns with the following names:
            - 'oncho_cases': the number of cases of onchocerciasis
            - 'opd_visits_pa': the number of outpatient department (OPD) visits per year per patient
            - 'severe_skin_visits': the number of visits per year per patient with severe skin disease
            - 'other_skin': the number of patients with other skin diseases
            - 'oncho_vision_visits': the number of visits per year per patient with onchocerciasis-associated eye disease
            - 'oae_cases': the number of cases of onchocerciasis-associated eye disease
            - 'oae_visits_pa': the number of visits per year per patient with onchocerciasis-associated eye disease
            - 'mf_baseline': the microfilarial load baseline value used in the simulations

            Example
            -------
            >>> import pandas as pd
            >>> simulated_data = {
                    'oncho_cases': [100, 200, 300],
                    'opd_visits_pa': [2, 2, 2],
                    'severe_skin_visits': [1, 2, 3],
                    'other_skin': [50, 100, 150],
                    'oncho_vision_visits': [2, 3, 4],
                    'oae_cases': [10, 20, 30],
                    'oae_visits_pa': [1, 1, 1],
                    'mf_baseline': 0.1
                }
            >>> simulated_df = pd.DataFrame(simulated_data)
            >>> total_annual_visits_2()
            (520.0, 120.0, 6, 225.0, 30, 139.99999999999997)
            """
            non_specific = simulated_df["oncho_cases"] * simulated_df["opd_visits_pa"]
            severe_skin = simulated_df["severe_skin_visits"]
            other_skin = simulated_df["other_skin"] * 1.5
            poor_vision = simulated_df["oncho_vision_visits"]
            oae_cases = simulated_df["oae_cases"] * simulated_df["oae_visits_pa"]
            total_visits = non_specific + severe_skin + other_skin + poor_vision + oae_cases
            return total_visits, non_specific, severe_skin, other_skin, poor_vision, oae_cases
        
        mf_baseline = mf_baseline/100
        
        
        def total_annual_visits_2():
            """
            Calculates the total annual number of visits required for different health conditions in a given population.

            This function uses a set of simulated data stored in the `simulated_df` DataFrame to estimate the total number of annual visits required for the following health conditions: non-specific cases, severe skin disease, other skin diseases, poor vision, and onchocerciasis-associated eye disease (OAE).

            Returns
            -------
            tuple
                A tuple with six elements:
                - total_visits: the total annual number of visits required for all health conditions
                - non_specific: the annual number of visits required for non-specific cases
                - severe_skin: the annual number of visits required for severe skin disease
                - other_skin: the annual number of visits required for other skin diseases
                - poor_vision: the annual number of visits required for poor vision
                - oae_cases: the annual number of visits required for OAE cases

            Notes
            -----
            The `simulated_df` DataFrame should contain columns with the following names:
            - 'oncho_cases': the number of cases of onchocerciasis
            - 'opd_visits_pa': the number of outpatient department (OPD) visits per year per patient
            - 'severe_skin_visits': the number of visits per year per patient with severe skin disease
            - 'other_skin': the number of patients with other skin diseases
            - 'oncho_vision_visits': the number of visits per year per patient with onchocerciasis-associated eye disease
            - 'oae_cases': the number of cases of onchocerciasis-associated eye disease
            - 'oae_visits_pa': the number of visits per year per patient with onchocerciasis-associated eye disease
            - 'mf_baseline': the microfilarial load baseline value used in the simulations

            Example
            -------
            >>> import pandas as pd
            >>> simulated_data = {
                    'oncho_cases': [100, 200, 300],
                    'opd_visits_pa': [2, 2, 2],
                    'severe_skin_visits': [1, 2, 3],
                    'other_skin': [50, 100, 150],
                    'oncho_vision_visits': [2, 3, 4],
                    'oae_cases': [10, 20, 30],
                    'oae_visits_pa': [1, 1, 1],
                    'mf_baseline': 0.1
                }
            >>> simulated_df = pd.DataFrame(simulated_data)
            >>> total_annual_visits_2()
            (520.0, 120.0, 6, 225.0, 30, 139.99999999999997)
            """
            non_specific = simulated_df["oncho_cases"] * simulated_df["opd_visits_pa"] * mf_baseline
            severe_skin = simulated_df["severe_skin_visits"] * mf_baseline
            other_skin = simulated_df["other_skin"] * 1.5 * mf_baseline
            poor_vision = simulated_df["oncho_vision_visits"] * mf_baseline
            oae_cases = simulated_df["oae_cases"] * simulated_df["oae_visits_pa"] * mf_baseline
            total_visits = non_specific + severe_skin + other_skin + poor_vision + oae_cases
            return total_visits, non_specific, severe_skin, other_skin, poor_vision, oae_cases

        def mean_ci(x):
            """
            Calculates the mean and 95% confidence interval of a given array of values.

            Parameters
            ----------
            x : numpy.ndarray or list
                An array or list of numeric values to be analyzed.

            Returns
            -------
            tuple
                A tuple with three elements:
                - ov_mean: the mean value of the input array
                - ov_lower: the lower bound of the 95% confidence interval
                - ov_upper: the upper bound of the 95% confidence interval

            Example
            -------
            >>> import numpy as np
            >>> data = np.array([2, 4, 6, 8, 10])
            >>> mean_ci(data)
            (6.0, 2.2, 9.8)
            """
            ov_upper = np.quantile(x, .975)
            ov_lower = np.quantile(x, .025)
            ov_mean = np.mean(x)
            return ov_mean, ov_lower, ov_upper

        mean_personnel_time = simulated_df["consultation_time"].mean()

        

        def ov_patients(func1, func2, baseline=1):
            """
            Summary function for health system impacts with and without MDA
            """
            ov_opd_narrative = f"""The estimated number of onchocerciasis cases at this level is: 
                    {func1(simulated_df["oncho_cases"])[0]* baseline:,.0f}[{func1(simulated_df["oncho_cases"])[1] * baseline:,.0f}, {mean_ci(simulated_df["oncho_cases"])[2] * baseline:,.0f}]. 
                        This assumes that 4% of the target population is affected. See the estimated annual clinic visits in the table below. We assume that no patients are admitted in hospital.
                        The OPD costs are based on WHO average costs for {country} which were estimated at {mean_opd_visit_cost:,.2f} USD in 2010. These values have been adjusted
                        to 2021 figures using an inflation rate of 3.0%."""
            varnames = ["Total visits", "Others", "Severe skin", "Other skin", "Poor vision", "OV-associated epilepsy"]

            var_means = [func1(func2()[0])[0], func1(func2()[1])[0], func1(func2()[2])[0],
                        func1(func2()[3])[0], func1(func2()[4])[0], func1(func2()[5])[0]]
        
            var_min = [func1(func2()[0])[1], func1(func2()[1])[1], func1(func2()[2])[1],
                        func1(func2()[3])[1], func1(func2()[4])[1], func1(func2()[5])[1]]

            var_max = [func1(func2()[0])[2], func1(func2()[1])[2], func1(func2()[2])[2],
                        func1(func2()[3])[2], func1(func2()[4])[2], func1(func2()[5])[2]]
            ov_opd_summary = pd.DataFrame({"Indicators": varnames, "Mean": var_means, "Min": var_min, "Max": var_max})
            ov_opd_summary["Personnel time"] = ov_opd_summary["Mean"] * mean_personnel_time
            visit_cost = mean_opd_visit_cost * 1.03 ** 11 # Adjusted OPD costs for inflation
            ov_opd_summary["Annual costs"] = ov_opd_summary["Mean"] * visit_cost
            
            personnel_time_narrative = f"""This translates to approximately {ov_opd_summary.loc[0, 'Personnel time']:,.0f} clinical 
                                        hours spent on onchocerciasis management annually at this level. 
                                        These personnel hours can be reallocated to other service areas."""
            return ov_opd_narrative, ov_opd_summary, personnel_time_narrative

        ov_opd_expander_1 = st.expander(translate_text("Summary of out-patient visits and health system costs without MDA"))

        with ov_opd_expander_1:
            st.write(translate_text(ov_patients(mean_ci, total_annual_visits_1)[0]))
            df1_sum = ov_patients(mean_ci, total_annual_visits_1)[1]
            col1, col2 = st.columns([2,1])
            with col1:
                st.dataframe(df1_sum.style.format({"Mean": "{:,.0f}", "Min": "{:,.0f}", 
                                                    "Max": "{:,.0f}", "Personnel time":"{:,.0f}",
                                                    "Annual costs": "${:,.2f}"}))
            with col2:   
                fig, ax = plt.subplots(figsize=(5,5))
                ax.pie(df1_sum["Mean"][1:], labels=df1_sum["Indicators"][1:], textprops={'fontsize': 12}, startangle=0, autopct='%1.1f%%')
                #ax.set_title("Proportion of costs by OV complication")
                st.pyplot(fig)
                download_chart(fig)
            st.write(translate_text(ov_patients(mean_ci, total_annual_visits_1)[2]))
            
        ov_opd_expander_2 = st.expander(translate_text("Summary of out-patient visits and health system costs with MDA"))
        with ov_opd_expander_2:
            st.write(translate_text(ov_patients(mean_ci, total_annual_visits_2, baseline=mf_baseline)[0]))
            df2_sum = ov_patients(mean_ci, total_annual_visits_2, baseline=mf_baseline)[1]
            col1, col2 = st.columns([2,1])
            with col1:
                st.dataframe(df2_sum.style.format({"Mean": "{:,.0f}", "Min": "{:,.0f}", 
                                                    "Max": "{:,.0f}", "Personnel time":"{:,.0f}",
                                                    "Annual costs": "${:,.2f}"}))
            with col2:   
                fig, ax = plt.subplots(figsize=(5,5))
                ax.barh(np.arange(len(df2_sum["Indicators"][1:])), df2_sum["Mean"][1:], color="green")    
                ax.set_xticks(range(len(df2_sum["Indicators"])))     
                ax.set_yticklabels(df2_sum["Indicators"])
                ax.tick_params(axis="both", labelsize=12)
                for spine in ax.spines.values():
                    spine.set_color('none')
                st.pyplot(fig)
                download_chart(fig)
            st.write(translate_text(ov_patients(mean_ci, total_annual_visits_2, baseline= mf_baseline)[2]))

        @st.cache
        def yld_ov():
            """Summary for years lived with disability. Cross-sectional"""
            varnames = ["Total YLD",  "Severe skin", "Other skin", "Poor vision", "Blind", "OV-associated epilepsy", "Premature mortality"]
            var_means = [mean_ci(simulated_df["yld_severe_itch"])[0], mean_ci(simulated_df["yld_other_skin"])[0], mean_ci(simulated_df["yld_low_vision"])[0],
                            mean_ci(simulated_df["yld_blind"])[0], mean_ci(simulated_df["yld_epilepsy"])[0], 
                            mean_ci(simulated_df["reduced_le"] * simulated_df["severe_itch_cases"])[0]]

            var_min =   [mean_ci(simulated_df["yld_severe_itch"])[1], mean_ci(simulated_df["yld_other_skin"])[1], mean_ci(simulated_df["yld_low_vision"])[1],
                            mean_ci(simulated_df["yld_blind"])[1], mean_ci(simulated_df["yld_epilepsy"])[1], 
                            mean_ci(simulated_df["reduced_le"] * simulated_df["severe_itch_cases"])[1]]

            var_max =   [mean_ci(simulated_df["yld_severe_itch"])[2], mean_ci(simulated_df["yld_other_skin"])[2], mean_ci(simulated_df["yld_low_vision"])[2],
                            mean_ci(simulated_df["yld_blind"])[2], mean_ci(simulated_df["yld_epilepsy"])[2], 
                            mean_ci(simulated_df["reduced_le"] * simulated_df["severe_itch_cases"])[2]]
            var_means.insert(0, sum(var_means))
            var_min.insert(0, sum(var_min))
            var_max.insert(0, sum(var_max)) 

            var_mean_mda = [x * mf_baseline for x in var_means]
            var_min_mda = [x * mf_baseline for x in var_min]
            var_max_mda = [x * mf_baseline for x in var_max]

            yld_summary = pd.DataFrame({
                "Indicators": varnames, "Mean no MDA": var_means, "Min. no MDA": var_min, "Max no MDA": var_max,
                "Mean MDA": var_mean_mda, "Min MDA": var_min_mda, "Max MDA": var_max_mda, 
            })
            non_specific = f"""Infected persons who do not have severe complications like blindness
                                also have general ill-health that reduces their quality of life and productivity.
                                We do not have data on this so, do not include this in the calculations.
                                Interpretation: Given the current status of the program at this level, approximately
                                <font color='blue'>*{var_means[0] - var_mean_mda[0]:,.0f}*</font> YLD are averted annually."""
            return yld_summary, non_specific
        
        with st.expander(translate_text("Years lived with disability summary")):
            st.dataframe(yld_ov()[0].style.format({"Mean no MDA": "{:,.1f}", "Min. no MDA": "{:,.1f}", "Max no MDA": "{:,.1f}",
                "Mean MDA": "{:,.1f}", "Min MDA": "{:,.1f}", "Max MDA":"{:,.1f}"}))
            st.write(translate_text(yld_ov()[1]), unsafe_allow_html=True)

        with st.expander(translate_text("Cost per DALY")):
            ov_dalys = yld_ov()[0]["Mean no MDA"].iloc[0] - yld_ov()[0]["Mean MDA"].iloc[0]
            gdp_ppp_x = country_inputs["Annual_PPP(Int$)"][country_inputs["Country"]==country].values[0]
            tot_ov_mda = mda_cost * pop_req_mda
            
            st.write(translate_text(f"""
                Assuming an average delivered MDA dose of {mda_cost} USD, and an MDA target population of {pop_req_mda:,.0f}, 
                the average cost per DALY averted is {tot_ov_mda/ov_dalys:,.2f} USD in {country}. This value is cost effective whether one uses the
                controversial GDP-based threshold of {gdp_ppp_x} USD or the alternative threshold of USD {cea_threshold():,.0f} using the approach 
                by [Woods et al (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5193154/)."""))
            
        with st.expander(translate_text("Economic Productivity")):

            st.write(translate_text(f"""<em>Patient level:</em> We estimated that approximately {simulated_df["patient_time"].mean():,.0f} productive person-years p.a. would have been lost at this level
                because of onchocerciasis had there been no MDA programmes. 
                With MDA, this declined to {(1 - simulated_df["morbidity_reduction_mda"].mean()) * simulated_df["patient_time"].mean():,.0f} 
                productive person-years p.a. lost. We assume that persons with chronic conditions e.g., leopard skin have long-term effects that cannot be reduced by MDA.
                This translates to productivity losses of {daily_wage()[0].values[0] * simulated_df["patient_time"].mean() * 262:,.0f} USD p.a. if we use inequality adjusted wages, 
                and {gdp_ppp_x * simulated_df["patient_time"].mean():,.0f} USD if we use GDP-based mean daily wages in the hypothetical absence of MDA. With current MDA programmes,
                this translates to productivity losses of {daily_wage()[0].values[0] * (1 - simulated_df["morbidity_reduction_mda"].mean()) * simulated_df["patient_time"].mean() * 262:,.0f} USD p.a. if we use inequality adjusted wages, 
                and {gdp_ppp_x * (1 - simulated_df["morbidity_reduction_mda"].mean()) * simulated_df["patient_time"].mean():,.0f} USD if we use GDP-based mean daily wages. """), unsafe_allow_html=True)
            
            st.write(translate_text(f"""<em>Caretaker level:</em> The average caretaker productivity loss at this level is {simulated_df["caretaker_time"].mean():,.0f} person-years p.a.
                The caretaker productivity losses are limited to people who are blind or have low vision and are therefore conservative. This assume
                {simulated_df["oncho_vision"].mean():,.0f} poor vision and {simulated_df["oncho_blind"].mean():,.0f} blind cases at this level.
                We assume that low vision and blindness are irreversible. This translates to {daily_wage()[0].values[0] * simulated_df["caretaker_time"].mean() * 262:,.0f} USD 
                p.a. if we use inequality adjusted wages, and {gdp_ppp_x * simulated_df["caretaker_time"].mean():,.0f} USD if we use GDP-based mean daily wages.
                The inequality-adjusted wages are estimated based on the earnings of the lowest quintile."""), unsafe_allow_html=True)   
            
        with st.expander(translate_text("Incremental cost benefit ratios")):
        
            def simulate_icers(n_simulations: int =1000, gdp: Optional[float] = None) -> pd.DataFrame:
                """
                Simulates ICERs (incremental cost-effectiveness ratios) for a given country and onchocerciasis control strategy.

                Parameters
                ----------
                n_simulations : int, optional
                    The number of simulations to run. Default is 10000.
                gdp : float, optional
                    The country's Gross Domestic Product (GDP) in international dollars (PPP). If not provided, a default value of 3000 is used.

                Returns
                -------
                pd.DataFrame
                    A DataFrame with columns for the simulated change in costs, MDA (mass drug administration) costs, change in effects, ICERs,
                    alternative ICERs, willingness-to-pay (WTP) thresholds, and probabilities of achieving each WTP threshold.

                Notes
                -----
                This function uses the `gamma_simulator` function to simulate gamma-distributed costs, and the `numpy.random.triangular`
                function to simulate triangular-distributed MDA costs and effects. The ICERs are calculated as the ratio of change in costs
                to change in effects, and the alternative ICERs are calculated as the ratio of MDA costs to change in effects. The WTP
                thresholds are generated as a linear range from 0 to 3 times the GDP of the country, and the probabilities of achieving
                each threshold are calculated as the proportion of ICERs below the threshold.
                """
                mean_effect = df1_sum.iat[0,1] - df2_sum.iat[0, 1]
                min_effect = df1_sum.iat[0,2] - df2_sum.iat[0, 2]
                max_effect = df1_sum.iat[0,3] - df2_sum.iat[0, 3]
                mean_cost = mean_opd_visit_cost * mean_effect
                min_cost = mean_opd_visit_cost * min_effect
                max_cost = mean_opd_visit_cost * max_effect
                costs = np.array(gamma_simulator(mean_cost, min_cost, max_cost))
                mda_sim = np.random.triangular(0.78 * oncho_cases, 1.23 * oncho_cases, 4.1* oncho_cases, size=n_simulations)
                effects = np.random.triangular( min_effect, mean_effect, max_effect, size=n_simulations)
                
                try:
                    gdp = country_inputs["Annual_PPP(Int$)"][country_inputs["Country"]==country].values[0]
                except Exception:
                    gdp = 3000
                icers = [c/e if e != 0 else np.nan for c,e in zip(costs, effects)]
                icer_mda = [c/e if e != 0 else np.nan for c, e in zip(mda_sim, effects)]
                wtp_thresholds = np.linspace(0, gdp * 3, n_simulations)
                probs = [np.mean(icers < t) for t in wtp_thresholds]
                wtp =[*range(n_simulations)]
                df = pd.DataFrame({
                    "Change in costs": costs,
                    "MDA SIM": mda_sim,
                    "Change in effects": effects,
                    "ICERs": icers,
                    "ICER2": icer_mda,
                    "WTP": wtp,
                    "Probs": probs
                })

                return df
            st.write(simulate_icers())
            source = simulate_icers()
            
            def plot_icers(source: pd.DataFrame):
                """
                Plots the Incremental Cost-Effectiveness Ratio (ICER) plane for a given data source.

                Parameters
                ----------
                source : pandas.DataFrame
                    The input data source containing the following columns:
                    - 'Change in costs': the change in costs compared to the comparator
                    - 'Change in effects': the change in health effects compared to the comparator
                    - 'ICER2': the ICER values to be used for color-coding the points in the scatter plot

                Returns
                -------
                altair.vegalite.v4.api.Chart
                    An Altair chart object that displays the ICER plane with the following elements:
                    - a scatter plot of the ICER values, with size and color encoding for the data points
                    - a cost-effectiveness acceptability curve (CEAC) represented by a dashed line
                    - a willingness-to-pay (WTP) threshold line represented by a solid red line
                    - a horizontal line at y=0 and a vertical line at x=0 to indicate the axes

                Example
                -------
                >>> source = icers(costs, effects, daly)
                >>> plot_icers(source)
                """

                scatter = alt.Chart(source).mark_circle(size=50, opacity=0.8).encode(
                y='Change in costs',
                x='Change in effects',
                color=alt.Color('ICER2', scale=alt.Scale(scheme='viridis')),
                tooltip=['ICER2']
                ).properties(
                    width=600,
                    height=600
                )
                
                ceac = alt.Chart(source).mark_line(color='black', strokeDash=[10, 5]).encode(
                    x='WTP',
                    y='ICER2'
                )
    
                wtp_slope = cea_threshold()
                max_wtp = source["Change in effects"].max()
                wtp_x = np.linspace(-max_wtp, max_wtp, 1000)
                wtp_y = wtp_x * wtp_slope
                temp_df = pd.DataFrame({
                    "wtp_x": wtp_x,
                    "wtp_y": wtp_y
                })

                line_plot = alt.Chart(temp_df).mark_line(color="red").encode(
                    x = "wtp_x",
                    y = "wtp_y"
                )
                # Add a horizontal line at y=0
                hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[10,10],color='black').encode(y='y')

                # Add a vertical line at x=0
                vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(strokeDash=[10,10], color='black').encode(x='x')
                icer_plane =  (scatter + ceac + line_plot + hline + vline).properties(
                    title='ICER plane'
                )
                return icer_plane
            
            st.write(plot_icers(source))

#==============================================================================================================================
    with tabs[4]:
        with st.expander(translate_text("Model technical inputs")):
            st.dataframe(oncho_data_technical.style.format({"Value": "{:,.2f}"}))   
            
    with tabs[5]:
        with st.expander(translate_text("Contact us")): 
            contact_form = f"""
                <form action="https://formsubmit.co/obierochieng@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder={translate_text("Your name")} required>
                    <input type="email" name="email" placeholder={translate_text("Your email")} required>
                    <input type="text" name="_honey" style="display:none">
                    <input type="hidden" name="_cc" value="ocu9@cdc.gov">
                    <textarea name="message" placeholder={translate_text("Details of your problem")}></textarea>
                    <button type={translate_text("submit")}>Send Information</button>
                </form>
            """   
            st.markdown(contact_form, unsafe_allow_html=True)

            def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            local_css("style/style.css")
