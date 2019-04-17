import numpy as np
import pandas as pd


class load_data:

    def __init__(self, path = '/Users/raphaellederman/Desktop/Text_clean/Data/essays.csv'):
        self.path = path

    def run(self):
        data_essays = pd.read_csv(self.path, encoding = "ISO-8859-1")
        data_essays['cEXT'] = np.where(data_essays['cEXT']=='y', 1, 0)
        data_essays['cNEU'] = np.where(data_essays['cNEU']=='y', 1, 0)
        data_essays['cAGR'] = np.where(data_essays['cAGR']=='y', 1, 0)
        data_essays['cCON'] = np.where(data_essays['cCON']=='y', 1, 0)
        data_essays['cOPN'] = np.where(data_essays['cOPN']=='y', 1, 0)
        X_essays = data_essays['TEXT'].tolist()
        y_essays = data_essays[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]
        labels = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
        return X_essays, y_essays, labels

