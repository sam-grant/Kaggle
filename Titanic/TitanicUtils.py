class SurvivalRate:
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