Tasks:

1. use glob for file operations
2. look up varity of facial recognition using opencv
3. Are uploads allowed? like xml resource files?

# How To Install
Execute the following:
    
    Set up conda environment.
    conda install -C conda-forge dlib
    pip install -r requirements.txt
    
# Strategies

    1. Problem? There is a potential problem in data leakage. The same person is in more than one video. If we split between test and validation,
    the system might learn to recognize a face base on the fact that 
