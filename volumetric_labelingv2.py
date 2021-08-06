#### IMPORT LIBRARIES
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
from magicgui import magicgui

# # import UI for stack selection
import tkinter as tk
from tkinter import filedialog

# %gui qt5
import os

#### PROCESSING FUNCTIONS
# GLOBAL VARIABLES
global VOLUME
VOLUME= None
global Z_MASK
Z_MASK = 1
global NEURON_MASK
NEURON_MASK = None

def adaptive_local_threshold(image, block_size):
    # adaptive_local_threshold is a function that takes in an image and applies an odd-integer block size
    # kernel (or filter) and thresholds based on local spatial information.

    return filters.threshold_local(image, block_size)

def gamma_level(image, gamma):
    # gamma_level is a function that takes in an image and changes the contrast by scaling the image
    # by a factor "gamma".
    return exposure.adjust_gamma(image, gamma)

def global_threshold_method(image, Threshold_Method):
    # global_threshold_method is a function that allows a user to choose what kind of method to binarize
    # an image to create a mask. For a given method, a threshold will be calculated and returns a binarized
    # image.
    if Threshold_Method == 'None':
        pass

    if Threshold_Method == 'Isodata':
        thresh = filters.threshold_isodata(image) # calculate threshold using isodata method
    if Threshold_Method == 'Li':
        thresh = filters.threshold_li(image) # calculate threshold using isodata method
    if Threshold_Method == 'Mean':
        thresh = filters.threshold_mean(image) # calculate threshold using isodata method
    if Threshold_Method == 'Minimum':
        thresh = filters.threshold_minimum(image)
    if Threshold_Method == 'Otsu':
        thresh = filters.threshold_otsu(image)
    if Threshold_Method == 'Yen':
        thresh = filters.threshold_yen(image)
    if Threshold_Method == 'Triangle':
        thresh = filters.threshold_triangle(image)
    else:
        thresh = 0

    tmp_img = image.copy()
    tmp_img[tmp_img<=thresh]=0
    return binary_labels(tmp_img)

def despeckle_filter(image, filter_method, radius):
    if filter_method == 'Erosion':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        eroded = morphology.erosion(tmp_img, footprint)
        return eroded

    if filter_method == 'Dilation':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        dilated = morphology.dilation(tmp_img, footprint)
        return dilated

    if filter_method == 'Opening':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        opened = morphology.opening(tmp_img, footprint)
        return opened

    if filter_method == 'Closing':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        closed = morphology.closing(tmp_img, footprint)
        return closed

def returnMask(mask):
    global Z_MASK
    Z_MASK = mask
    return Z_MASK

def binary_labels(image):
    # function binary_labels labels the entire neuron and entries of the Image = 2. Later, 2 => 'Dendrites'
    labels_array = image.copy()

    neuron = labels_array * Z_MASK
    auto = labels_array * (1-Z_MASK)
    labels_array[neuron > 0] = 2
    labels_array[auto > 0] = 6

    labels_array = labels_array.astype('int')

    return labels_array


#### MAIN WIDGET/PROGRAM HERE

@magicgui(
    image = {'label': 'Image'},
    gamma = {"widget_type": "FloatSlider", 'max': 5},
    block_size = {"widget_type": "SpinBox", 'label': 'Block Size:', 'min': 1, 'max': 20},
    threshold_method = {"choices": ['None','Isodata', 'Li', 'Mean', 'Minimum', 'Otsu', 'Triangle', 'Yen']},
    speckle_method = {"choices": ['None','Erosion', 'Dilation', 'Opening', 'Closing']},
    radius = {"widget_type": "SpinBox", 'max': 10, 'label': 'Radius'},
    layout = 'vertical'
 )

def threshold_widget(image: ImageData,
                     gamma = 1,
                     block_size = 3,
                     threshold_method = 'None',
                     speckle_method = 'None',
                     radius = 1
                     ) -> LayerDataTuple:
    #function threshold_widget does a series of image processing and thresholding to binarize the image and make a label

    if image is not None:
        # adjust the gamma levelz
        label_img = gamma_level(image, gamma)

        # go through the stack and perform the local threshold
        for curr_stack in range(np.shape(label_img)[0]):
            label_img[curr_stack] = adaptive_local_threshold(label_img[curr_stack], block_size)

        # finally do a global threshold to calculate optimized value to make it black/white
        label_img = global_threshold_method(label_img, threshold_method)
        label_img = despeckle_filter(label_img, speckle_method, radius)

        return (label_img, {'name': 'neuron_label'}, 'labels')


#### WIDGET FOR PROCESSING IMAGE AND SHOWING THE PROCESSED IMAGE BEFORE SEGMENTATION
# from magicgui import widgets

@magicgui(
    image = {'label': 'Image'},
    filter_method = {"choices": ['None','median', 'gaussian', 'bilateral', 'TV']},
    value_slider = {"widget_type": "FloatSlider", 'max': 4, 'label': 'None'},
    layout = 'vertical'
 )
def smoothen_filter(image: ImageData,
                  filter_method = 'None',
                  value_slider = 1) -> LayerDataTuple:
    # filter_widget is a function that takes an input image and selects a filter method
    # for denoising an image.
    # Returns an IMAGE layer.

    if image is not None:
        stack_size = np.shape(image)[0]
        if filter_method == 'median': # use a median filter and go through the entire stack to apply the filter
            tmp_img = image.copy()
            for curr_stack in range(stack_size):
                tmp_img[curr_stack] = filters.median(tmp_img[curr_stack], morphology.disk(value_slider))
            return (tmp_img, {'name': 'smoothened_image'}, 'image')

        if filter_method == 'gaussian': # use a gaussian filter
            tmp_img = image.copy()
            tmp_img = filters.gaussian(tmp_img, sigma = value_slider)
            return (tmp_img, {'name': 'smoothened_image'}, 'image')

        if filter_method == 'bilateral': # use a bilateral filter
            tmp_img = image.copy()
            for curr_stack in range(stack_size):
                tmp_img[curr_stack] = restoration.denoise_bilateral(tmp_img[curr_stack], sigma_spatial = value_slider)

            return (tmp_img, {'name': 'smoothened_image'}, 'image')

        if filter_method == 'TV': # using a total-variation (TV) denoising filter
            tmp_img = image.copy()
            for curr_stack in range(stack_size):
                tmp_img[curr_stack] = restoration.denoise_tv_chambolle(tmp_img[curr_stack], weight = value_slider)
            return (tmp_img, {'name': 'smoothened_image'}, 'image')

@smoothen_filter.filter_method.changed.connect
def change_label(event):
    # change_label function is written to change the label of the FloatSlider widget
    # such that the user won't be confused as to what metric is being used.

    if event.value == 'median':
        smoothen_filter.value_slider.label = 'Pixel Radius'
    if event.value == 'gaussian':
        smoothen_filter.value_slider.label = 'sigma'
    if event.value == 'bilateral':
        smoothen_filter.value_slider.label = 'sigma_spatial'
    if event.value == 'TV':
        smoothen_filter.value_slider.label = 'weight'
        smoothen_filter.value_slider.max = 1
        smoothen_filter.value_slider.value = 0

#####################################################################################\

### Widget for using shapes to get segmentation
from skimage.draw import polygon

@magicgui(
    call_button = 'Generate Neuron Volume',
    layout = 'vertical'
)
def generate_neuron_volume():

    shape_mask = z_projection_viewer.layers[1].data[0]

    px_coord = np.zeros(z_projection_viewer.layers[0].data.shape, dtype = np.uint8) # initialize map of rows and columns

    rr, cc = polygon(shape_mask[:,0], shape_mask[:,1]) # get the rows and columns from polygon shape
    px_coord[rr, cc] = 1 # set all the rows and columns in the matrix as 1

    returnMask(px_coord)

    print("Mask Shape: ", Z_MASK.shape)

    z_projection_viewer.window.add_dock_widget(generate_neuron_volume) # undock the mask generator widget
    z_projection_viewer.close()

    viewer = napari.Viewer()
    viewer.add_image(VOLUME, name = 'Neuron', blending='additive')
    viewer.window.add_dock_widget(smoothen_filter, name = 'Smoothen Filter')
    viewer.window.add_dock_widget(threshold_widget, name = 'Thresholding')
    viewer.window.add_dock_widget(save_layer, name = 'Save Files')
    # napari.run(max_loop_level = 2)
    
    
#####################################################################################

#### WIDGET FOR SAVING LAYER AS H5 FILE

@magicgui(
    call_button = 'Save Layer',
    file_picker = {"widget_type": 'FileEdit', 'value': 'N/A', 'mode': 'd'})
def save_layer(image: ImageData, label: Labels, file_picker = 'N/A'):
    folder_name = file_picker
    file_str = os.path.splitext(os.path.basename(file_path))[0]
    h5_name = file_str + '.h5'
    full_dir = os.path.join(folder_name, h5_name)

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

#####################################################################################

# file_path = os.path.join(neuron_dir,neuron_file)

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

if os.path.splitext(file_path)[1] == '.h5':
    viewer = napari.Viewer()
    edit_labels = h5py.File(file_path, 'r+')
    # load in the image and label and add to viewer
    neuron_image = np.array(edit_labels['project_data'].get('raw_image'))
    label_layer = np.array(edit_labels['project_data'].get('label'))
    edit_labels.close()

    viewer.add_image(neuron_image, name = 'Neuron')
    viewer.add_labels(label_layer, name = 'Neuron_label')
    viewer.window.add_dock_widget(smoothen_filter, name = 'Smoothen Filter')
    viewer.window.add_dock_widget(threshold_widget, name = 'Thresholding')
    viewer.window.add_dock_widget(save_layer, name = 'Save Files')
    napari.run()
    
else:
    z_projection_made = False
    neuron_image = skimage.io.imread(file_path)

    # GLOBAL VARIABLES
    VOLUME = neuron_image.copy()
    NEURON_MASK = np.zeros_like(VOLUME)
    z_projection_viewer = napari.Viewer()
    z_projection_viewer.window.add_dock_widget(generate_neuron_volume)
    # create a z projection of neuron volume max pixel intensities
    z_projection = neuron_image.copy()
    z_projection = np.max(neuron_image, axis=0)
    z_projection_viewer.add_image(z_projection, name = 'Neuron Projection')
    z_projection_viewer.add_shapes()
    napari.run()
