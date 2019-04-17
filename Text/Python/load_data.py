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

    def visualize(self):

        # Visualization of text length vs. label
        for label in self.labels:
            g = sns.FacetGrid(data=self.data_essays, col=label)
            g.map(plt.hist, 'text length', bins=50)
        plt.show()

        for i, label in enumerate(self.labels):
            plt.figure(i)
            sns.boxplot(x=label, y='text length', data=self.data_essays)
        plt.show()

        # Visualization of the most frequent words
        complete_corpus = ' '.join(self.X_essays)
        words = tokenize.word_tokenize(complete_corpus)
        fdist = FreqDist(words)
        fdist.plot(40)





