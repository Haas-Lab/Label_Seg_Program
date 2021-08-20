from typing import Any
import h5py
import numpy as np

# import all skimage related stuff
import skimage.io
from skimage.measure import label
from skimage import filters, exposure, restoration
from skimage import morphology

# import all napari related stuff
import napari
from napari.types import ImageData, LabelsData, LayerDataTuple, ShapesData
from napari.layers import Image, Layer, Labels, Shapes
from magicgui.backends._qtpy import show_file_dialog
from magicgui import magicgui

import os

def save_layer(image: ImageData, label: Labels, file_picker: any, path):
    file_str = os.path.splitext(os.path.basename(path))[0]
    h5_name = file_str + '.h5'
    full_dir = os.path.join(file_picker, h5_name)

    if os.path.isfile(full_dir): # if the file exists and layer needs to be overwritten
        hf = h5py.File(full_dir, 'r+')
        new_label = label.data # new labelled data
        curr_label = hf['project_data']['label']
        # print(curr_label)
        curr_label[:] = new_label
        hf.close()
        # check if changes were properly made:
        hf = h5py.File(full_dir, 'r+')
        print(np.allclose(hf['project_data']['label'], new_label))
        hf.close()

    else: # for if the file doesn't exist yet, create the h5 file
        # Dictionary for label ints
        label_dict = {
                'Background' : 0,
                'Soma' : 1,
                'Dendrite' : 2,
                'Filopodia' : 3,
                'Axon' : 4,
                'Growth Cone' : 5,
                'Autofluorescence' : 6,
                'Melanocyte' : 7,
                'Noise' : 8,
        }
        label_dict = str(label_dict) # make dictionary as string in order to save into h5 file. Use ast library to return it back into dict
            # initialize HDF5 file
        hf =  h5py.File(full_dir, 'a')
        grp = hf.create_group("project_data")

        # save the raw image
        try:
            grp.create_dataset('raw_image', data = image.data)
            print('Successfully Saved Raw Data')
        except:
            print('Saving Raw Data Unsuccessful')        
        # save the label
        try:
            grp.create_dataset('label', data = label.data)
            print('Successfully saved Labeled Data')
        except:
            print('Saving Labeled Data Unsuccessful') 
        # save the associated dictionary 
        try: 
            grp.create_dataset('label_dict', data=label_dict)
            print('Succesfully Saved Labeled Dictionary')
        except:
            print('Saving Label Dictionary Unsuccessful')  

        hf.close()
