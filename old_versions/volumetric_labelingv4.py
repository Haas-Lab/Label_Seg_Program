#### IMPORT LIBRARIES
import h5py
import numpy as np
from h5py._hl.base import Empty
from numpy.lib import type_check
from scipy import ndimage
import sys
import os
path = os.getcwd()
sys.path.append(path)

# import all skimage related stuff
import skimage.io
from skimage.measure import label
from skimage import filters, exposure, restoration, registration
from skimage import morphology

# import processing functions
from processing.processing_functions import *

# import machine learning modules
import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.inferers.inferer import SliceInferer, SlidingWindowInferer


# import all napari related stuff
import napari
from napari.types import ImageData, LabelsData, LayerDataTuple, ShapesData
from napari.layers import Image, Layer, Labels, Shapes
from magicgui import magicgui
from magicgui import widgets

# %gui qt5
# # import UI for stack selection
from magicgui.backends._qtpy import show_file_dialog

# import all subprocess to run ilastik
import subprocess

from aicsimageio import AICSImage, imread
from skimage.io import imsave

#### PROCESSING FUNCTIONS
# GLOBAL VARIABLES
global VOLUME
VOLUME= None
global Z_MASK
Z_MASK = 1
global NEURON
NEURON = [0,0]
global NOISE
NOISE = [0,0]
global NEURON_LABLED
NEURON_LABLED = False
global NOISE_LABLED
NOISE_LABLED = False
global COMPLETED_LABEL
COMPLETED_LABEL = None
global EDIT_EXISTING_LABEL
EDIT_EXISTING_LABEL = False
global AI_PREDICTION
AI_PREDICTION = None

def adaptive_local_threshold(image, block_size):
    # adaptive_local_threshold is a function that takes in an image and applies an odd-integer block size
    # kernel (or filter) and thresholds based on local spatial information.

    return filters.threshold_local(image, block_size)

def adaptive_threshold_3d(volume, block_size, offset=0, filter_sigma=None):
    if block_size % 2 == 0:
        raise ValueError("Block size should be an odd number")

    if filter_sigma is not None:
        local_mean = ndimage.gaussian_filter(volume, filter_sigma)
    else:
        half_block_size = block_size // 2
        padded_volume = np.pad(volume, ((half_block_size, half_block_size),) * 3, mode='reflect')
        local_mean = ndimage.uniform_filter(padded_volume, size=block_size, mode='constant')

        # Remove padding from local_mean
        local_mean = local_mean[
            half_block_size:local_mean.shape[0] - half_block_size,
            half_block_size:local_mean.shape[1] - half_block_size,
            half_block_size:local_mean.shape[2] - half_block_size
        ]

    return (volume > local_mean + offset).astype(np.uint8)

def gamma_level(image, gamma):
    # gamma_level is a function that takes in an image and changes the contrast by scaling the image
    # by a factor "gamma".
    return exposure.adjust_gamma(image, gamma)

def get_adjusted_image(layer: Image):
    # get the shape of the data
    # normalize data by contrast lims
    normalized_data = (layer.data - layer.contrast_limits[0]) / (layer.contrast_limits[1] - layer.contrast_limits[0])
    
    # clip the values to [0, 1] range
    clipped_data = np.clip(normalized_data, 0, 1)
    # apply gamma correction
    gamma_corrected_data = clipped_data ** layer.gamma
    
    # get the original min and max values of the data type
    min_val, max_val = layer.contrast_limits_range

    # scale the data by the original range
    scaled_data = gamma_corrected_data * (max_val - min_val) + min_val
    
    return scaled_data

def model_predict(image, mode, spatial_dim):
    path = os.getcwd()
    image = image*Z_MASK

    # load yaml file required for getting configurations of the model
    yaml_file = f'{path}/production_models/{mode}_{spatial_dim}_ResUNet.yml'
    import yaml
    from yaml.loader import SafeLoader
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=SafeLoader)

    lateral_steps = config['DATASET']['lateral_steps']
    axial_steps = config['DATASET']['axial_steps']
    batch_size = config['DATASET']['batch_size']
    input_chnl = 1
    output_chnl = 4
    norm_type = config['MODEL']['norm']
    dropout = config['MODEL']['dropout']

    # define the UNET model
    model = UNet(spatial_dims=config['MODEL']['spatial_dim'], 
                in_channels = input_chnl,
                out_channels = output_chnl,
                channels = config['MODEL']['channel_layers'],
                strides=config['MODEL']['strides'],
                num_res_units=config['MODEL']['num_res_units'],
                norm = norm_type,
                dropout = dropout)
    model_path = f'{path}/production_models/{mode}_{spatial_dim}_ResUNet.pth'
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    if config['MODEL']['spatial_dim'] == 3:
        patch_size = (axial_steps, lateral_steps, lateral_steps)
        inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=batch_size, progress=True)
    else:
        patch_size = (lateral_steps, lateral_steps)
        inferer = SliceInferer(roi_size=patch_size, sw_batch_size=batch_size, spatial_dim = 0, progress=True)

    raw_transform = transforms.Compose([MinMaxScalerVectorized()])
    processed_img_dataset = WholeVolumeDataset(raw_img=image,
                                               raw_transform=raw_transform)

    processed_img, _ = next(iter(processed_img_dataset))
    processed_img = torch.unsqueeze(processed_img, dim = 0)

    with torch.no_grad():
        output = inferer(inputs = processed_img, network=model)
        # print(output.shape)
        probabilities = torch.softmax(output,1)
        # print(probabilities.shape)
        pred = to_numpy(torch.argmax(probabilities, 1)).astype(np.int16)

    # soma category is inferenced for only "Neuron" => change all the soma labels (1) to dendrite labels (2)
    if len(np.unique(pred)) == 2:
        pred[pred==1] = 2
        return pred, len(np.unique(pred))+1
    else:
        return pred, len(np.unique(pred))
    

def global_threshold_method(image, Threshold_Method, labelee):
    # global_threshold_method is a function that allows a user to choose what kind of method to binarize
    # an image to create a mask. For a given method, a threshold will be calculated and returns a binarized
    # image.
    if Threshold_Method == 'None':
        return image
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
        print("threshold value: ", thresh)
    elif Threshold_Method == 'Yen':
        thresh = filters.threshold_yen(image)
    elif Threshold_Method == 'Triangle':
        thresh = filters.threshold_triangle(image)
    else:
        thresh = 0

    tmp_img = image.copy()
    tmp_img[tmp_img<=thresh]=0
    tmp_img = binary_labels(tmp_img, labelee)
    
    return tmp_img

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

def returnMask(mask,vol_shape):
    global Z_MASK
    tmp_mask = np.zeros(vol_shape)
    for z in range(vol_shape[0]):
        tmp_mask[z,...] = mask
    Z_MASK = tmp_mask
    return Z_MASK

def binary_labels(image, labelee):
    # function binary_labels labels the neuron and entries of the Image = 2. Later, 2 => 'Dendrites', 
    # returns either the neuron labeled or noise labeled
    labels_array = image.copy()

    neuron = labels_array * Z_MASK
    auto = labels_array * (1-Z_MASK)
    
    if labelee == 'neuron':
        labels_array[neuron > 0] = 2
        labels_array[auto > 0] = 0
        
        
    if labelee == 'noise': 
        labels_array[neuron > 0] = 0   
        labels_array[auto > 0] = 9
        
    labels_array = labels_array.astype('int')
    return labels_array

# Offset correction
def returnOffsetCorr(image):
    if np.min(image)<0:
        image = image - np.min(image)
    return image

#### MAIN WIDGET/PROGRAM HERE

@magicgui(
    image = {'label': 'Image'},
    # raw_gamma = {"widget_type": "FloatSlider", 'max': 5},
    # ai_gamma = {"widget_type": "FloatSlider", 'max': 5},
    threshold_method = {"choices": ['None','Local','Isodata', 'Li', 'Mean', 'Minimum', 'Otsu', 'Triangle', 'Yen']},
    block_size = {"widget_type": "SpinBox", 'label': 'Block Size:', 'step': 2, 'min': 1, 'max': 100, 'visible':False},
    small_object_count = {"widget_type": "SpinBox", 'label': 'Min Cluster Size:', 'min': 0, 'max': 200},
    AI_Assist = {"choices": ['None', 'Neuron', 'Soma+Dendrite', 'Soma+Dendrite+Filo']},
    allow_rerun={"label": "Allow rerun", "nullable": True},
    AI_model_dim = {"choices": ["2D","3D"]},
    speckle_method = {"choices": ['None','Erosion', 'Dilation', 'Opening', 'Closing']},
    radius = {"widget_type": "SpinBox", 'max': 10, 'label': 'Radius'},
    affected = {"choices": ['neuron', 'noise']},
    layout = 'vertical'
 )
def threshold_neuron_widget(
                     image: Image,
                     block_size = 1,
                     threshold_method = 'Otsu',
                     small_object_count = 0,
                     AI_Assist = 'None',
                     allow_rerun: bool = True,
                     include_autofluorescence = False,
                     AI_model_dim = "2D",
                     speckle_method = 'None',
                     radius = 1,
                     affected = 'neuron',
                     ) -> LayerDataTuple:
    #function threshold_widget does a series of image processing and thresholding to binarize the neuron parts of the 
    #image and make a label while only affecting the noise or the neuron mask
    print(affected)
    if image is not None:
        listOfGlobals = globals()
        # adjust the gamma levelz
        # processed_image = gamma_level(image, raw_gamma)
        processed_image = get_adjusted_image(image)
        processed_image = np.rint(processed_image)

        # finally do a global threshold to calculate optimized value to make it black/white

        if affected == 'neuron':
            if threshold_method != "None":
                if threshold_method == "Local":
                    label_img = np.zeros(processed_image.shape)
                    # for curr_stack in range(np.shape(label_img)[0]):
                    #     local_thresh = adaptive_local_threshold(processed_image[curr_stack], block_size)
                    #     label_img[curr_stack] = processed_image[curr_stack] > local_thresh
                    label_img = adaptive_threshold_3d(processed_image,block_size)
                    label_img = binary_labels(label_img, affected)
                    # remove small objects
                    label_img = morphology.label(label_img)
                    label_img = morphology.remove_small_objects(label_img, min_size=small_object_count)
                    label_img[label_img > 0] = 2
                else:
                    label_img = np.zeros(processed_image.shape)
                    label_img = global_threshold_method(processed_image, threshold_method, affected)
                    # remove small objects
                    label_img = morphology.label(label_img)
                    label_img = morphology.remove_small_objects(label_img, min_size=small_object_count)
                    label_img[label_img > 0] = 2
                
            else:
                label_img = np.zeros(processed_image.shape).astype(np.int8)

            if AI_Assist != 'None':
                if AI_Assist == "Soma+Dendrite":
                    num_classes = 4
                if AI_Assist == "Soma+Dendrite+Filo":
                    num_classes = 5

                if allow_rerun == True:
                    
                    ai_image = processed_image
                    # if AI_model_dim == "2D": # use the 2D equivalent to the 3D model i.e., 2D unet for slice inference
                    model_img, num_classes = model_predict(ai_image, AI_Assist, AI_model_dim)
                    model_img = to_categorical(model_img,num_classes=num_classes).astype(bool)

                    # clean up any false positives or extremely small objects identified as dendrites
                    model_img[...,2] = morphology.remove_small_objects(model_img[...,2], min_size=small_object_count)
                    model_img = np.argmax(model_img,axis=-1) 

                    if include_autofluorescence == False:
                        # find the max class and make everywhere that has autofluorescence to be 0
                        autofluor = model_img.max()
                        model_img[model_img==autofluor] = 0
                    listOfGlobals['AI_PREDICTION'] = model_img
                    threshold_neuron_widget.allow_rerun.value = False #uncheck the box for rerunning
                    if threshold_method == "None":
                        label_img = model_img
                    else:
                        model_img = to_categorical(model_img,num_classes=num_classes).astype(bool)
                        tmp_label_img = to_categorical(label_img, num_classes=num_classes).astype(bool)
                        tmp_categorical = np.zeros(model_img.shape)
                        for i in range(num_classes):
                            if i == 0:
                                tmp_categorical[...,i] = model_img[...,i]*tmp_label_img[...,i] # "AND" background
                            else:
                                tmp_categorical[...,i] = model_img[...,i]+tmp_label_img[...,i] # "OR" every other label

                        label_img = np.argmax(tmp_categorical,axis=-1)
                else:
                    model_img = listOfGlobals['AI_PREDICTION']
                    if threshold_method == "None":
                        label_img = model_img
                    else:
                        model_img = to_categorical(model_img,num_classes=num_classes).astype(bool)
                        tmp_label_img = to_categorical(label_img, num_classes=num_classes).astype(bool)
                        tmp_categorical = np.zeros(model_img.shape)
                        for i in range(num_classes):
                            if i == 0:
                                tmp_categorical[...,i] = model_img[...,i]*tmp_label_img[...,i] # "AND" background
                            else:
                                tmp_categorical[...,i] = model_img[...,i]+tmp_label_img[...,i] # "OR" every other label

                        label_img = np.argmax(tmp_categorical,axis=-1)

            if speckle_method != "None":
                label_img = despeckle_filter(label_img, speckle_method, radius)
            
            listOfGlobals['NEURON'] = label_img.copy()
            listOfGlobals['NEURON_LABLED'] = True
        
        if affected == 'noise':
            label_img = np.zeros(processed_image.shape)
            label_img = global_threshold_method(processed_image, threshold_method, 'noise')
            if speckle_method != "None":
                label_img = despeckle_filter(label_img, speckle_method, radius)
            label_img = morphology.label(label_img)
            label_img = morphology.remove_small_objects(label_img, min_size=small_object_count)
            label_img[label_img > 0] = 9
            listOfGlobals['NOISE'] = label_img.copy()
            listOfGlobals['NOISE_LABLED'] = True


        global ENTIRE_IMAGE
        if EDIT_EXISTING_LABEL == True:
            # ENTIRE_IMAGE = listOfGlobals['NOISE']
            if listOfGlobals['NOISE_LABLED']:
                ENTIRE_IMAGE = np.add(listOfGlobals['COMPLETED_LABEL'],listOfGlobals['NOISE'])
                return (ENTIRE_IMAGE, {'name': 'Neuron_label'}, 'labels')
            else:
                return (COMPLETED_LABEL, {'name': 'Neuron_label'}, 'labels')

        else:
            if listOfGlobals['NEURON_LABLED']:
                if listOfGlobals['NOISE_LABLED']:
                    ENTIRE_IMAGE = np.add(listOfGlobals['NEURON'], listOfGlobals['NOISE'])
                    return (ENTIRE_IMAGE, {'name': 'neuron_label'}, 'labels')
                else:
                    ENTIRE_IMAGE = listOfGlobals['NEURON']
                    return (ENTIRE_IMAGE, {'name': 'neuron_label'}, 'labels')
            else:
                ENTIRE_IMAGE = listOfGlobals['NOISE']
                return (ENTIRE_IMAGE, {'name': 'neuron_label'}, 'labels')



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

    if event == 'median':
        smoothen_filter.value_slider.label = 'Pixel Radius'
    if event == 'gaussian':
        smoothen_filter.value_slider.label = 'sigma'
    if event == 'bilateral':
        smoothen_filter.value_slider.label = 'sigma_spatial'
    if event == 'TV':
        smoothen_filter.value_slider.label = 'weight'
        smoothen_filter.value_slider.max = 1
        smoothen_filter.value_slider.value = 0

@threshold_neuron_widget.threshold_method.changed.connect
def on_method_changed(event):
    if event == "Local":
        threshold_neuron_widget.block_size.visible = True
    else:
        threshold_neuron_widget.block_size.visible = False

@magicgui(
    call_button = 'Run Ilastik',
    layout = 'vertical'
 )
def run_ilastik() -> LayerDataTuple:
    # Img Base Name
    IMAGE = show_file_dialog(caption = 'choose your image')
    BASENAME = os.path.basename(IMAGE)

    # Ilastik Install Location
    #ILASTIK_LOC = show_file_dialog(caption = 'choose where ilastik applciation is')
    ILASTIK_LOC = '/home/peter/Applications/ilastik-1.3.3post3-Linux/run_ilastik.sh'
    # Neuron Segmentation
    # ILASTIK_PRO_NEURON = show_file_dialog(caption = 'choose ilastik file for neuron segmentation',
    #                                       filter = '.ilp')

    # Neuronal Subdomain Classifier
    #ILASTIK_PRO_SUB = show_file_dialog(caption = 'choose ilastik file for neuron subdomain classification')

    ILASTIK_PRO_NEURON = "/home/peter/Applications/DynaROI/Ilastik_Tectal_Neuron_Autocontext/tectal_neuron_auto.ilp"
    ILASTIK_PRO_SUB = '/home/peter/Applications/DynaROI/Subdomain_Training_Stacks/Subdomain_Classifier.ilp'

    # STEP 1: Run the pixel classifier for general structure of the neuron

    '''

    print('running ilastik classifier')
    launch_args = [ILASTIK_LOC,
               '--headless',
               '--project='+ILASTIK_PRO_NEURON,
               '--export_source=probabilities stage 2',
               IMAGE,
               '--output_filename_format=results/{nickname}_neuron_seg.h5'
               ]
    subprocess.run(launch_args)
    print('finished ilastik classification')

    pixel_classifier = h5py.File("results/"+BASENAME[:-4]+"_neuron_seg.h5", 'r') # read in pixel classifier results
    neuron_data = skimage.io.imread(IMAGE) # read raw neuron image

    image = neuron_data[0,0,:,:,:]
    image = image[0,:,:,:]
    masks = pixel_classifier['exported_data']
    neuron_seg = masks[:,:,:,1].copy()
    neuron_seg[neuron_seg<.85]=0 # only accept probabilities that are less than 85%

    masked_neuron = neuron_seg * image

    '''

    # STEP 2: Run the pixel classifier for sub regions of the neuron i.e. soma, dendrites, etc.

    print('Running Ilastik Classifier for Subregions')
    launch_args = [ILASTIK_LOC,
            '--headless',
            '--project='+ILASTIK_PRO_NEURON,
            '--export_source=probabilities stage 2',
            IMAGE,
            '--output_filename_format=results/{nickname}_neuron_seg.h5'
                       ]

    subprocess.run(launch_args)
    pixel_classifier = h5py.File("results/"+BASENAME[:-4]+"_neuron_seg.h5", 'r')
    data1 = imread(IMAGE)
    base_image = imread(IMAGE)

    image = data1[0,0,:,:,:]
    image = image[0,:,:,:]
    masks = pixel_classifier['exported_data']
    neuron_seg = masks[:,:,:,1].copy()
    neuron_seg[neuron_seg<.85]=0

    masked_neuron = neuron_seg * image
    # Save the segemented image to use in subdomain classification
    imsave('results/neuron_seg_'+BASENAME, masked_neuron)

    launch_args = [ILASTIK_LOC,
                   '--headless',
                   '--project='+ILASTIK_PRO_SUB,
                   '--export_source=probabilities',
                   'results/neuron_seg_'+BASENAME,
                   '--output_filename_format=results/{nickname}_sub_seg.h5'
                   ]



    subprocess.run(launch_args)

    print('Finished running classifier')

    pixel_classifier = h5py.File("results/neuron_seg_"+BASENAME[:-4]+"_sub_seg.h5", 'r')

    masks = pixel_classifier['exported_data']

    background = masks[:,:,:,0].copy()
    background[background<.65]=0

    # extract soma from mask
    soma = masks[:,:,:,1].copy()
    soma[soma<.90]=0
    soma[soma > 0] = 1 #

    # extract dendrite from mask
    dendrites =  masks[:,:,:,2].copy()
    dendrites[dendrites<.85]=0
    dendrites[dendrites > 0] = 2

    # extract filopodia's
    filopodia =  masks[:,:,:,3].copy()
    filopodia[filopodia<.8]=0
    filopodia[filopodia > 0] = 3

    neuron_mask = np.zeros_like(dendrites)
    neuron_mask[dendrites==2] = 2
    neuron_mask[soma==1] = 1
    neuron_mask[filopodia==3] = 3

    neuron_mask = neuron_mask.astype('int')
    imsave('results/neuron_labels_'+BASENAME+'.tif', neuron_mask)

    return (neuron_mask, {'name': 'neuron_mask'}, 'labels')

#####################################################################################

### Widget for using trained ML model for segmentation

# @magicgui(
#     call_button = 'Generate Mask',
#     model = {"choices": ['ResUNet','BasicUNet', 'gaussian', 'bilateral', 'TV']},
#     layout = 'vertical'
#  )

#####################################################################################\


from skimage.draw import polygon
@magicgui(
    image = {'label': 'Image'},
    call_button = "Create Z-Projection",
    layout = 'vertical'
)
def make_z_projection(image: ImageData):
    z_projection = np.max(image, axis=0)
    viewer.add_image(z_projection, name = 'Neuron Projection')
    viewer.add_shapes()

@magicgui(
    call_button = "Generate Mask",
    layout = 'vertical'
)
def generate_mask():

    shape_mask = viewer.layers[-1].data[0]
    px_coord = np.zeros(viewer.layers[-2].data.shape, dtype = np.uint8) # initialize map of rows and columns

    rr, cc = polygon(shape_mask[:,0], shape_mask[:,1]) # get the rows and columns from polygon shape
    px_coord[rr, cc] = 1 # set all the rows and columns in the matrix as 1
    returnMask(px_coord, VOLUME.shape)

@magicgui(
    call_button = "Reset Autofluorescence",
    layout = "vertical"
)
def reset_autofluorescence():
    # find where all the autofluorescence is and delete them from the image
    COMPLETED_LABEL[COMPLETED_LABEL==6] = 0 # reset all autofluorescent labels
    COMPLETED_LABEL[COMPLETED_LABEL>7] = 0 # all labels greater than 7 (melanocytes)

    return (COMPLETED_LABEL, {'name': 'Neuron_label'}, 'labels')


### Widget for using shapes to get segmentation
@magicgui(
    call_button = 'Generate Neuron Volume',
    layout = 'vertical'
)
def generate_neuron_volume():

    shape_mask = z_projection_viewer.layers[1].data[0]

    px_coord = np.zeros(z_projection_viewer.layers[0].data.shape, dtype = np.uint8) # initialize map of rows and columns

    rr, cc = polygon(shape_mask[:,0], shape_mask[:,1]) # get the rows and columns from polygon shape
    px_coord[rr, cc] = 1 # set all the rows and columns in the matrix as 1

    returnMask(px_coord, VOLUME.shape)

    print("Mask Shape: ", Z_MASK.shape)

    # z_projection_viewer.window.add_dock_widget(generate_neuron_volume) # undock the mask generator widget
    z_projection_viewer.close()

    viewer = napari.Viewer()
    viewer.add_image(VOLUME, name = 'Neuron', blending='translucent_no_depth')
    viewer.window.add_dock_widget(smoothen_filter, name = 'Smoothen Filter')
    viewer.window.add_dock_widget(threshold_neuron_widget, name = 'Thresholding')
    viewer.window.add_dock_widget(importPreviousLabels, name = 'Import From Last Time Point')
    #viewer.window.add_dock_widget(run_ilastik, name = 'Ilastik Classifier (BETA)')
    viewer.window.add_dock_widget(save_layer, name = 'Save Files')
    # napari.run(max_loop_level = 2)

#####################################################################################

#### WIDGET FOR SAVING LAYER AS H5 FILE

@magicgui(
    call_button = 'Save Layer',
    file_picker = {"widget_type": 'FileEdit', 'value': 'N/A', 'mode': 'd'},
    Type_Name = {"widget_type": 'LineEdit', 'value': 'Enter Your Name'},
    Fluoro_Name = {"choices": ['EGFP-F','jYCaMP1', 'GluSnFR']},
)
def save_layer(image: ImageData,
                label: Labels,
                file_picker = 'N/A',
                Type_Name = 'Enter Your Name',
                Fluoro_Name= 'EGFP-F',
                is_complete = False):

    folder_name = file_picker
    labeler = Type_Name
    # type(labeler)
    # type(Fluoro_Name)
    file_str = os.path.splitext(os.path.basename(file_path))[0]
    h5_name = file_str + '.h5'
    full_dir = os.path.join(folder_name, h5_name)

    if is_complete:
        completion_cond = 'True'
    else:
        completion_cond = 'False'

    label_dict = {
                'Background' : 0,
                'Soma' : 1,
                'Dendrite' : 2,
                'Filopodia' : 3,
                'Axon' : 4,
                'Growth Cone' : 5,
                'Autofluorescence' : 6,
                'Melanocyte' : 7,
                'Noise' : 9,
        }

    if os.path.isfile(full_dir): # if the file exists and layer needs to be overwritten

        hf = h5py.File(full_dir, 'r+')

        project_data = hf['project_data'] # access the project_data group

        raw_image_data = hf['project_data']['raw_image'] # read into raw_image data

        # overwrite label data
        new_label = label.data
        curr_label = hf['project_data']['label'] # read into label data
        curr_label[:] = new_label
        print('Overwritten previous labels with current labels!')

        # make metadata for the entire project data set

        if project_data.attrs.items() == (): #create metadata if there aren't any items
            project_data.attrs['completeness'] = completion_cond
            project_data.attrs['labeler'] = labeler

        # make metadata for raw_image
        if raw_image_data.attrs.items() == ():
            # create fluorophore metadata
            raw_image_data.attrs['fluorophore'] = Fluoro_Name
            print('Created Metadata - Fluorophore: ' + Fluoro_Name)

        # make metadata for label data
        if curr_label.attrs.items() == (): # check if there are any attributes in the label data: if it returns empty list then start creating metadata
            print('Creating metadata set...')

            # create metadata for subdomains for the label
            for k in label_dict.keys():
                curr_label.attrs[k] = label_dict[k]
                print('Created Metadata - subdomain: ' + k)

        print('Updated Metadata or have Created new metadata only: ')
        print(list(project_data.attrs.items()))
        print(list(raw_image_data.attrs.items()))
        # print(list(curr_label.attrs.items()))
        hf.close()

        # check if changes were properly made:
        hf = h5py.File(full_dir, 'r+')
        if np.allclose(hf['project_data']['label'], new_label):
            print('Label Successfully Overwritten: Current label is now saved into project_data group.')
        hf.close()

    else: # for if the file doesn't exist yet, create the h5 file

        # initialize HDF5 file
        h5_name = file_str + "_"+ labeler.encode().hex()[:8] + '.h5'
        full_dir = os.path.join(folder_name, h5_name)
        hf =  h5py.File(full_dir, 'a')

        grp = hf.create_group("project_data")
        grp.attrs.create('completeness', completion_cond)
        print('Creating metadata set...')
        # create labeler metadata
        grp.attrs['labeler'] = labeler
        print('Created Metadata - Labeler: ' + labeler)

        # save the raw image
        try:
            im_data = grp.create_dataset('raw_image', data = image.data)
            print('Successfully Saved Raw Data')
            im_data.attrs['fluorophore'] = Fluoro_Name
            print('Created Metadata - Fluorophore: ' + Fluoro_Name)
        except:
            print('Saving Raw Data Unsuccessful')

        # save the label and corresponding metadata
        try:
            lab_data = grp.create_dataset('label', data = label.data)
            print('Successfully saved Labeled Data')

            # create metadata for subdomains for the label
            for k in label_dict.keys():
                lab_data.attrs[k] = label_dict[k]
                print('Created Metadata - subdomain: ' + k)
        except:
            print('Saving Labeled Data Unsuccessful')

        print('Created New Dataset and Following are the metadata saved: ')
        print(list(grp.attrs.items()))
        print(list(im_data.attrs.items()))
        print(list(lab_data.attrs.items()))

        hf.close()

#####################################################################################
@magicgui(
    file_picker = {"widget_type": 'FileEdit', 'value': 'N/A' }
)
def importPreviousLabels(image: ImageData, file_picker = 'N/A')-> LayerDataTuple:
    previous_time = h5py.File(file_picker, 'r+')
    # load in the image and label and add to viewer
    last_image = np.array(previous_time['project_data'].get('raw_image'))
    last_label = np.array(previous_time['project_data'].get('label'))
    previous_time.close()
    upsampleFactor=1
    shifts = registration.phase_cross_correlation(image, last_image, upsample_factor=upsampleFactor, return_error=False)
    print(shifts)
    new_labels = ndimage.shift(last_label, shifts)
    new_labels = new_labels.astype('int')

    return (new_labels, {'name': 'neuron_label'}, 'labels')

#####################################################################################
# file_path = os.path.join(neuron_dir,neuron_file)

file_path = show_file_dialog()

if os.path.splitext(file_path)[1] == '.h5':
    EDIT_EXISTING_LABEL = True

    viewer = napari.Viewer()
    edit_labels = h5py.File(file_path, 'r+')
    # load in the image and label and add to viewer
    neuron_image = np.array(edit_labels['project_data'].get('raw_image'))
    VOLUME = neuron_image.copy()

    label_layer = np.array(edit_labels['project_data'].get('label'))
    COMPLETED_LABEL = label_layer
    edit_labels.close()

    reviewer_edit_widget = widgets.Container(widgets=[make_z_projection, generate_mask, reset_autofluorescence])

    viewer.add_image(neuron_image, name = 'Neuron')
    viewer.add_labels(label_layer, name = 'Neuron_label')
    viewer.window.add_dock_widget(smoothen_filter, name = 'Smoothen Filter')
    viewer.window.add_dock_widget(threshold_neuron_widget, name = 'Thresholding')
    viewer.window.add_dock_widget(reviewer_edit_widget, name = "Add Noise to Existing Labels")
    viewer.window.add_dock_widget(save_layer, name = 'Save Files')
    napari.run()

else:
    z_projection_made = False
    neuron_image = skimage.io.imread(file_path)
    if len(neuron_image.shape) == 4:
        neuron_image = np.average(neuron_image,axis=1)

    # GLOBAL VARIABLES
    VOLUME = neuron_image.copy()
    # NEURON_MASK = np.zeros_like(VOLUME)
    z_projection_viewer = napari.Viewer()
    z_projection_viewer.window.add_dock_widget(generate_neuron_volume)
    # create a z projection of neuron volume max pixel intensities
    z_projection = neuron_image.copy()
    z_projection = np.max(neuron_image, axis=0)
    z_projection_viewer.add_image(z_projection, name = 'Neuron Projection')
    z_projection_viewer.add_shapes()
    napari.run()
