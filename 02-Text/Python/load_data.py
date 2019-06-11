import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import *

class load_data:

    def __init__(self, path = '/Users/raphaellederman/Desktop/Fil_Rouge_Local/Text/Data/essays.csv'):
        self.path = path

    def run(self):
        self.data_essays = pd.read_csv(self.path, encoding = "ISO-8859-1")
        self.data_essays['cEXT'] = np.where(self.data_essays['cEXT']=='y', 1, 0)
        self.data_essays['cNEU'] = np.where(self.data_essays['cNEU']=='y', 1, 0)
        self.data_essays['cAGR'] = np.where(self.data_essays['cAGR']=='y', 1, 0)
        self.data_essays['cCON'] = np.where(self.data_essays['cCON']=='y', 1, 0)
        self.data_essays['cOPN'] = np.where(self.data_essays['cOPN']=='y', 1, 0)
        self.X_essays = self.data_essays['TEXT'].tolist()
        self.y_essays = self.data_essays[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]
        self.labels = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
        return self.data_essays, self.X_essays, self.y_essays, self.labels





