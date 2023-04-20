# Label_Seg_Program
Label Segmentation Program used for semi-automated semantic segmentation annotation of 3D neuronal volumes.

## Installation:
1. Make sure to have miniconda(arm64 apple silicon)/anaconda(window,macos intel, linux) and homebrew (arm64 apple silicon) installed on your machine.
2. Open the terminal/cmd and change directory to the location you want to have the labeling program installed in.
3. copy and paste each of the following lines
```
git clone git@github.com:Haas-Lab/Label_Seg_Program.git
cd Label_Seg_Program
conda create -y -n labeling_env -c conda-forge python=3.9 --file requirements.txt
conda activate labeling_env
```
4. Download deep learning models from google drive:
```
python setup.py
```
5. Launch the program: 
```
python volumetric_labelingv4.py
```
