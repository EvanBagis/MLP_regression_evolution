from setuptools import setup

setup(name='MLP_regression_evolution',
      version='0.1',
      description='genetic algorithm for ANN hyperparameter tuning',
      url='https://github.com/EvanBagis/neuro-evolution-regression',
      author='Evan Bagis',
      author_email='evanbagis@gmail.com',
      license='MIT',
      packages=['MLP_regression_evolution'],
      install_requires=['tqdm', 'numpy', 'logger'],
      zip_safe=False)
