#### IMPORT LIBRARIES
import h5py
import numpy as np
## TESTing
# from saving import save_layer
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
    for curr_stack in range(np.shape(image)[0]):
            image[curr_stack] = filters.threshold_local(image[curr_stack], block_size)
    return image


def global_threshold_method(image, Threshold_Method):
    # global_threshold_method is a function that allows a user to choose what kind of method to binarize
    # an image to create a mask. For a given method, a threshold will be calculated and returns a binarized
    # image.
    if Threshold_Method == 'None':
        pass
    elif Threshold_Method == 'Isodata':
        thresh = filters.threshold_isodata(image) # calculate threshold using isodata method
    elif Threshold_Method == 'Li':
        thresh = filters.threshold_li(image) # calculate threshold using isodata method
    elif Threshold_Method == 'Mean':
        thresh = filters.threshold_mean(image) # calculate threshold using isodata method
    elif Threshold_Method == 'Minimum':
        thresh = filters.threshold_minimum(image)
    elif Threshold_Method == 'Otsu':
        thresh = filters.threshold_otsu(image)
    elif Threshold_Method == 'Yen':
        thresh = filters.threshold_yen(image)
    elif Threshold_Method == 'Triangle':
        thresh = filters.threshold_triangle(image)
    else:
        thresh = 0

    tmp_img = image.copy()
    tmp_img[tmp_img<=thresh]=0
    return binary_labels(tmp_img)

def despeckle_filter(image, filter_method, radius):
    if filter_method == 'None':
        return image

    elif filter_method == 'Erosion':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        eroded = morphology.erosion(tmp_img, footprint)
        return eroded

    elif filter_method == 'Dilation':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        dilated = morphology.dilation(tmp_img, footprint)
        return dilated

    elif filter_method == 'Opening':
        tmp_img = image.copy()
        footprint = morphology.disk(radius)
        footprint = footprint[None,:,:]
        opened = morphology.opening(tmp_img, footprint)
        return opened

    elif filter_method == 'Closing':
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
        # Adjust the gamma levels to improve contrast for thresholding
        label_img = exposure.adjust_gamma(image, gamma)

        # Perform the local threshold
        label_img = adaptive_local_threshold(label_img, block_size)

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

######################################################################################

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
    viewer.window.add_dock_widget(save_layer(image = neuron_image,label=label_layer, file_picker=HELP, path=file_path), name = 'Save Files')
    # napari.run(max_loop_level = 2)
    
    
#####################################################################################

#### WIDGET FOR SAVING LAYER AS H5 FILE

from saving import save_layer
magicgui(
    function = save_layer,
    call_button = 'Save Layer',
    file_picker = {"widget_type": 'FileEdit', 'value': 'N/A', 'mode': 'd'}
    )

#########################################################################################


file_path = show_file_dialog()

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
    viewer.window.add_dock_widget(save_layer(image = neuron_image,label=label_layer, file_picker=HELP, path=file_path), name = 'Save Files')
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
