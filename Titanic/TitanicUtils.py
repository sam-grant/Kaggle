class SurvivalRate:
  """Helper class to calculate surivial rates per param"""
  def __init__(self, data=None, printout=True, output=False):
    """ Initialize with data. """
    self.data = data
    self.printout = printout
    self.output = output
    self.out_ = {}  # Dictionary to store results
    
  def get_sex_rate(self):
    ''' Sex survival rate '''
    if self.data is None:
      print("No data provided.")
      return

    out_ = {}
    sex_ = list(set(self.data.Sex))  # Get unique sexes
    for sex in sex_:
      val_ = self.data.loc[self.data.Sex == sex, 'Survived']  # Select Survived column based on Sex
      rate = val_.mean()  # Calculate mean (survival rate)
      if self.printout:
        print(f'Surviving {sex}: {100*rate:.1f}%')
      out_[sex] = 100 * rate

    if self.output:
      return out_
    else:
      return
    
  def get_class_rate(self):
    ''' Class survival rate '''
    if self.data is None:
      print("No data provided.")
      return

    out_ = {}
    Pclass_ = list(set(self.data.Pclass))  # Get unique classes
    for Pclass in Pclass_:
      val_ = self.data.loc[self.data.Pclass == Pclass, 'Survived']
      rate = val_.mean()
      if self.printout:
        print(f'Survivors in class {Pclass}: {100*rate:.1f}%')
      out_[Pclass] = 100 * rate
        
    if self.output:
      return out_
    else:
      return
    
  def get_sibsp_rate(self):
    ''' Spouse/siblings survival rate '''
    if self.data is None:
      print("No data provided.")
      return

    out_ = {}
    SibSp_ = list(set(self.data.SibSp))  # Get unique number of siblings/spouse
    for SibSp in SibSp_:
      val_ = self.data.loc[self.data.SibSp == SibSp, 'Survived']
      rate = val_.mean()
      if self.printout:
        print(f'Survivors with {SibSp} spouse + siblings: {100*rate:.1f}%')
      out_[SibSp] = 100 * rate
        
    if self.output:
      return out_
    else:
      return
    
  def get_parch_rate(self):
    ''' Parents/children survival rate '''
    if self.data is None:
      print("No data provided.")
      return

    out_ = {}
    Parch_ = list(set(self.data.Parch))  # Get unique number of parents/children
    for Parch in Parch_:
      val_ = self.data.loc[self.data.Parch == Parch, 'Survived']
      rate = val_.mean()
      if self.printout:
        print(f'Survivors with {Parch} parents + children: {100*rate:.1f}%')
      out_[Parch] = 100 * rate
        
    if self.output:
      return out_
    else:
      return
    
# Imports for CompareResults()
import sys
sys.path.append('../Common')
from PlotUtils import Plot

import matplotlib.pyplot as plt
from scipy.stats import chi2
import pandas as pd 

class CompareResults():
    """Compare training dataset with model prediction"""
    def __init__(self, test_, train_, output_):
        """ Initialize with data. """
        self.test_ = test_
        self.train_ = train_
        self.output_ = output_

    def run(self):

        # Merge output with test dataset for comparison
        pred_ = pd.merge(self.output_, self.test_, on='PassengerId')
        display(pred_)

        # Training surival rate 
        srate_t = SurvivalRate(self.train_, printout=False, output=True)
        # Predicted surivial rate
        srate_p = SurvivalRate(pred_, printout=False, output=True)

        # Parameters to compare on 
        sex_ = { 'Train' : srate_t.get_sex_rate(), 'Prediction' : srate_p.get_sex_rate() } 
        class_ = { 'Train' : srate_t.get_class_rate(), 'Prediction' : srate_p.get_class_rate() } 
        sibsp_ = { 'Train' : srate_t.get_sibsp_rate(), 'Prediction' : srate_p.get_sibsp_rate() } 
        parch_ = { 'Train' : srate_t.get_parch_rate(), 'Prediction' : srate_p.get_parch_rate() } 

        # Make plots
        pl = Plot() 

        fig, ax1 = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

        pl.plot_bar_overlay(data_dict=sex_, xlabel='Sex', ylabel='Survival rate [%]', save=False, show=True, ax=ax1[0])
        pl.plot_bar_overlay(data_dict=class_, xlabel='PClass', save=False, show=True, ax=ax1[1])
        pl.plot_bar_overlay(data_dict=sibsp_, xlabel='SibSp', save=False, show=True, ax=ax1[2])
        pl.plot_bar_overlay(data_dict=parch_, xlabel='Parch', save=False, show=True, ax=ax1[3])

        # Rough comparison
        comparison = {}

        # Sex comparison (handling categories 'male' and 'female')
        comparison['Sex'] = {
            'male': sex_['Train'].get('male', 0) - sex_['Prediction'].get('male', 0),
            'female': sex_['Train'].get('female', 0) - sex_['Prediction'].get('female', 0)
        }

        # Pclass comparison (handling categories 1, 2, 3)
        comparison['Pclass'] = {
            1: class_['Train'].get(1, 0) - class_['Prediction'].get(1, 0),
            2: class_['Train'].get(2, 0) - class_['Prediction'].get(2, 0),
            3: class_['Train'].get(3, 0) - class_['Prediction'].get(3, 0),
        }

        # SibSp comparison (handling various values)
        comparison['SibSp'] = {
            0: sibsp_['Train'].get(0, 0) - sibsp_['Prediction'].get(0, 0),
            1: sibsp_['Train'].get(1, 0) - sibsp_['Prediction'].get(1, 0),
            2: sibsp_['Train'].get(2, 0) - sibsp_['Prediction'].get(2, 0),
            3: sibsp_['Train'].get(3, 0) - sibsp_['Prediction'].get(3, 0),
            4: sibsp_['Train'].get(4, 0) - sibsp_['Prediction'].get(4, 0),
            5: sibsp_['Train'].get(5, 0) - sibsp_['Prediction'].get(5, 0),
            8: sibsp_['Train'].get(8, 0) - sibsp_['Prediction'].get(8, 0),
        }

        # Parch comparison (handling various values)
        comparison['Parch'] = {
            0: parch_['Train'].get(0, 0) - parch_['Prediction'].get(0, 0),
            1: parch_['Train'].get(1, 0) - parch_['Prediction'].get(1, 0),
            2: parch_['Train'].get(2, 0) - parch_['Prediction'].get(2, 0),
            3: parch_['Train'].get(3, 0) - parch_['Prediction'].get(3, 0),
            4: parch_['Train'].get(4, 0) - parch_['Prediction'].get(4, 0),
            5: parch_['Train'].get(5, 0) - parch_['Prediction'].get(5, 0),
            6: parch_['Train'].get(6, 0) - parch_['Prediction'].get(6, 0),
            9: parch_['Train'].get(9, 0) - parch_['Prediction'].get(9, 0),
        }

        fig, ax2 = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

        # print(comparison['Sex'])
        pl.plot_bar(data_dict=comparison['Sex'], xlabel='Sex', ylabel=r'Train $-$ Prediction [%]', col='green', save=False, show=True, ax=ax2[0])
        pl.plot_bar(data_dict=comparison['Pclass'], xlabel='Pclass', col='green', save=False, show=True, ax=ax2[1])
        pl.plot_bar(data_dict=comparison['SibSp'], xlabel='SibSp', col='green', save=False, show=True, ax=ax2[2])
        pl.plot_bar(data_dict=comparison['Parch'], xlabel='Parch', col='green', save=False, show=True, ax=ax2[3])

        residuals = []
        for param in comparison.keys():
            for val in comparison[param].values():
                residuals.append(val)

        # print('residuals', residuals)
        pl.plot1D(
            residuals,
            nbins=50, xmin=-100, xmax=100, leg_pos='upper right',
            xlabel='Train $-$ Prediction [%]', ylabel='Counts', save=False)
        
        n_entries, mean, mean_err, std_dev, std_dev_err, underflows, overflows = pl.get_stats(residuals, xmin=-100, xmax=100)

        return std_dev, std_dev_err
        
        # print(f'Std Dev =  {std_dev:.1f}+/-{std_dev_err:.1f}%')

        # # Calculate chi-squared
        # chi_squared = np.sum(np.square(residuals))  # Sum of squared residuals
        # degrees_of_freedom = len(residuals) - 1  # Typically df = N - 1 for a single sample

        # # Output chi-squared statistic
        # print(f"Chi2: {chi_squared:.2f}")
        # print(f"Degrees of freedom: {degrees_of_freedom}")
        # print(f"Chi2/ndf: {chi_squared/degrees_of_freedom:.2f}")


        # # To get a p-value, we would need to use the chi-squared distribution:

        # p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)
        # print(f"P-value: {p_value:.4f}")