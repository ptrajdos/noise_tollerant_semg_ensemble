from setuptools import setup, find_packages
import platform

def numpy_version():
    if "armv" in platform.machine():
        return 'numpy==1.23.4'
    return 'numpy>=1.22.4'

def numba_version():
    if "armv" in platform.machine():
        return 'numba==0.56.4'

    return 'numba'

setup(
        name='dexterous_bioprosthesis_2021_raw_datasets_framework',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/noise_tollerant_semg_ensemble',
        description="Dataset creation framework. Instances are represented by raw signals objects",
        packages=find_packages(include=[
                'dexterous_bioprosthesis_2021_raw_datasets_framework',
                'dexterous_bioprosthesis_2021_raw_datasets_framework.*',
		'dexterous_bioprosthesis_2021_raw_datasets_framework_experiments'
                ]),
        install_requires=[ 
                'pandas',
                numpy_version(),
                'matplotlib',
                'scipy>=1.10.1',
                'liac-arff',
                'joblib',
                'scikit-learn==1.2.2',
                'tqdm',
                'joblib',
                'dtw-python',
                'Cython',
                'fastdtw',
                'tensorly',
                'pygad==2.18.0',
                'kneed',
                'librosa',
                'audiomentations', 
                'problexity',
                'mspca @git+https://github.com/ptrajdos/mspca.git@8020d725e243818129f4af3d67b5c7d4486c90f7',
                'pyts==0.13.0',
                'DESlib==0.3.5',
                numba_version(),
                'statsmodels==0.13.5',
                'PyWavelets==1.4.1',
                'kernelnb @ git+https://github.com/ptrajdos/KernelNB.git@c38c01a7bf440417852a3b656c4353fab3dc46c9',
                'pt_outlier_probability @ git+https://github.com/ptrajdos/ptOutlierProbability.git@4d137a12220612ed6078178a6cf54b4c98699d99',
                'EntropyHub==2.0',
                'ptranks @ git+https://github.com/ptrajdos/ptranks.git@7af07a0f3666f15b2794b078da440bcf89fae65b',
                'tabulate==0.9.0',
                'Jinja2==3.1.2',
                'seaborn==0.13.2',
                'seaborn',
                
        ],
        test_suite='test'
        )
