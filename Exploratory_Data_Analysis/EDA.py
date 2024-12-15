# this is program that implements an Exploratory Data Analysis of
# any dataset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

class Exploratory_Data_Analysis:
    def __init__(self, dataset):
        self.dataset = dataset
