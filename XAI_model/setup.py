from setuptools import setup, find_packages

setup(
    name='XAI_model',        
    version='0.1.0',                 
    packages=find_packages(),        
    install_requires=[               
        'tensorflow>=2.0.0',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'statsmodels'
    ],
    author='Yeseul Jeon',
    author_email='jeons9677@gmail.com',
    description='Deep Generative Modeling for Spatial and Network Images (XAI)',
    long_description=open('README.md').read(),   
    long_description_content_type='text/markdown',
    url='https://github.com/YOUR_USERNAME/XAI_model', 
    license='MIT',
    classifiers=[                  
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
)
