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

# Conda Management:

Update Conda:

    conda update -n base -c defaults conda
    
Create New Env:

    conda create --name python3.7.6 python=3.7.6
    
Activate conda:


Upgrade Pip

python -m pip install --upgrade pip
    
# How to install 

1. Start Miniconda with Admin priv.

2. conda activate python3.7.6

1. Install pytorch stuff:
    Windows:
    
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    
    MacOs:
    
    conda install pytorch torchvision -c pytorch

2. pip install -r requirements.txt


# How to run test

python -m pytest tests\services\TestFaceRecogService.py::TestFaceRecogService::test_mtcnn_get_many_face_recog -s

