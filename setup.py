from setuptools import find_packages, setup

setup(name='merlion',
      version='0.1',
      description='Machine Learning Package',
      url='https://github.com/thufirtan/merlion',
      author='KC Tan',
      author_email='kctan6@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'lightgbm', 'pandas', 'sklearn', 'category_encoders', 'bayesian-optimization', 'shap'
      ],
      zip_safe=False)