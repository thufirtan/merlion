from setuptools import find_packages, setup

setup(name='merlion',
      version='0.2',
      description='Machine Learning Package',
      author='KC Tan',
      author_email='kctan6@gmail.com',
      url = 'https://github.com/thufirtan/merlion',
      download_url = 'https://github.com/thufirtan/merlion/archive/v0.2.tar.gz',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'lightgbm', 'pandas', 'sklearn', 'category_encoders', 'optuna', 'shap'
      ],
      zip_safe=False)