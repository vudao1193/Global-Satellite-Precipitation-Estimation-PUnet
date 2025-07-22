###############################################
# PERSIANN-Unet (PUnet) model for global satellite precipitation estimation using infrared images
# by Phu Nguyen & Vu Dao at Center for Hydrometeorology & Remote Sensing at UC Irvine
# March 2025
###############################################
# Input is merg IR binary .gz file (2, 3298, 9896)

# Import necessary libraries
import cv2
import os
import numpy as np
import gzip
import struct
import glob

# Set the system run on CPUs, and filter out info and warning messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Input, Dropout
from tensorflow.keras.models import Model
from scipy.ndimage import uniform_filter
from cv2.ximgproc import guidedFilter
import concurrent.futures
import tempfile
from tqdm import tqdm
import imresize

###############################################

# Define paths for parameters and input/output data
PARAMETER_DIR = "./Parameters/"
MODEL_WEIGHT_DIR = "./Model weight/"
IR_DATA_DIR = "./IR_data/"
OUTPUT_DIR = "./PUnet_output/"

# Define start and end day (format: yyyymmddhh)
startday = 	2022010100  
endday = 	2022010105   

# Number of CPU cores for parallel processing
num_cpus = 6

###############################################
# Functions

# Function to define and load a U-Net model
# It takes input data and a set of weights to produce predictions
def load_model_pred(inp, weight):
    input_shape = (512, 1536, 2)
    input_data = Input(shape=input_shape, name='input_data')

    # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(input_data)
    conv1 = BatchNormalization()(conv1)  #
    conv1 = tf.keras.layers.ReLU()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)  #
    conv2 = tf.keras.layers.ReLU()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = tf.keras.layers.ReLU()(conv4)
    conv4 = Dropout(0.4)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up3 = UpSampling2D(size=(2, 2))(conv4)
    up3 = concatenate([conv3, up3], axis=-1)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=-1)
    conv6 = Conv2D(128, (3, 3), padding='same')(up2)
    conv6 = BatchNormalization()(conv6)  #
    conv6 = tf.keras.layers.ReLU()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up1 = UpSampling2D(size=(2, 2))(conv6)
    up1 = concatenate([conv1, up1], axis=-1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    # Output Layer
    output_data = Conv2D(1, (1, 1), activation='relu')(conv7)

    model = Model(inputs=input_data, outputs=output_data)
    model.set_weights([weight[f"weight_{i}"] for i in range(len(weight))])

    # Run the model on input
    pred =  model(inp, training=False)
    pred = np.array(pred[0, ..., 0])
    tf.keras.backend.clear_session()

    return pred

# Function to clean bad (NaN) pattern in IR images
def removebad(ir0):
    ir = np.copy(ir0)
    x = np.copy(ir0)

    # Identify values less than 0
    x = np.where(x < 0, -2.0, 0)
    # Create a 2D filter
    filter_size = (100, 200)
    x1 = uniform_filter(x, size=filter_size, mode='reflect')
    ir[x1 < -0.5] = -99
    return ir

# Function to fill missing (NaN) in IR images
def fill_missing(ir_nan, window_size=10, mode='reflect'):

    nan_mask = np.isnan(ir_nan).astype(np.float32)
    ir_filled_temp = np.where(nan_mask, 0, ir_nan)

    smoothed = uniform_filter(ir_filled_temp, size=window_size, mode=mode)
    normalization = uniform_filter(1 - nan_mask, size=window_size, mode=mode)
    normalization[normalization == 0] = np.nan

    filled_values = smoothed / normalization
    ir_filled = np.where(nan_mask == 1, filled_values, ir_nan)
    return ir_filled

# Function to apply a mapping function for bias correction
def apply_mapping_function(future_satellite_estimate, satellite_quantiles, observed_quantiles):
    # Create a mask indicating where interpolation should be performed (i.e., where future_satellite_estimate is non-zero)
    non_zero_mask = future_satellite_estimate > 0.001

    # Get the indices where the mask is True
    lat_indices, lon_indices = np.where(non_zero_mask)
    corrected_estimate = np.copy(future_satellite_estimate)

    # Interpolate along the first axis (quantiles) for each non-zero pixel
    for lat, lon in zip(lat_indices, lon_indices):
        # Extract quantile arrays for the current pixel
        sat_quantiles = satellite_quantiles[:, lat, lon]
        obs_quantiles = observed_quantiles[:, lat, lon]

        value = future_satellite_estimate[lat, lon]
        # Perform interpolation
        if value > sat_quantiles[-1]:
            # Extrapolation for values above the maximum satellite quantile
            slope = max((obs_quantiles[-1] - obs_quantiles[-2]) / (sat_quantiles[-1] - sat_quantiles[-2] + 1e-6), 1.0)
            corrected_estimate[lat, lon] = obs_quantiles[-1] + slope * (value - sat_quantiles[-1])
        else:
            # Interpolation for values within the satellite quantile range
            corrected_estimate[lat, lon] = np.interp(value, sat_quantiles, obs_quantiles)
    return corrected_estimate

# Function to apply multi-scale guided filtering for noise reduction and quality enhancement
def multi_scale_guided_filter(guide, target, base_radius, base_epsilon, scales):
    filtered = target.copy()
    for scale in scales:
        scaled_radius = base_radius * scale
        scaled_epsilon = base_epsilon * scale**2
        filtered = guidedFilter(guide, filtered, int(scaled_radius), scaled_epsilon)
    return filtered

# Mainfunction to process input data and produce output file
def process_file(file_ir):
    # Gunzip to read file first
    with gzip.open(file_ir, 'rb') as f_in:
        om = np.frombuffer(f_in.read(), dtype=np.uint8)
    # Reshape the array according to dimensions
    # Bin data is scaled by subtracting 75 from the real temperature value, need to +75 to back the real temperature
    # 1 file include two timestep at 15 and 45
    om = om.astype(np.float32).reshape((2, 3298, 9896))+75 
    # 255 is the missing value (or 330 after value has been unscaled)
    om[om==330]=np.nan
        
    # Processing IR
    max_ir = 300
    min_ir = 173
    n_row = 512
    n_col = 1536
    #Loop over 2 timestep of each IR file
    for i in range(2):
        ir0=om[i,...]
        timestep = '15' if i == 0 else '45'
        # Clean IR data  and resize IR to 1200x3600 for guided filtering and masking NaN 
        ir=removebad(ir0)
        ir[ir <min_ir] = np.nan
        # filling missing 2 time before and after resize to 1200x3600 
        ir = fill_missing(ir)
        ir = np.concatenate((ir[:, 4948:], ir[:, :4948]), axis=1)
        ir = imresize.imresize(ir, output_shape=[1200,3600],method = 'bilinear')
        ir = fill_missing(ir)
        mask = np.where(np.isnan(ir))

        # Create guided image for guided filtering
        ir1= ir.copy()
        ir1[ir1 > max_ir] = max_ir
        ir1[ir1 < min_ir] = np.nan
        ir1 = (max_ir - ir1) / (max_ir - min_ir) * 255.0
        ir1[ir1 < 0] = np.nan
        ir1[np.isnan(ir1)] = 0
        
        # Resize IR to 512x1536, and normalize values for Unet model
        ir2 = imresize.imresize(ir, output_shape=[n_row,n_col],method = 'bilinear')
        ir2[ir2 > max_ir] = max_ir
        ir2[ir2 < min_ir] = np.nan
        # Perform the intensity transformation
        ir2 = (max_ir-ir2)/(max_ir-min_ir)
        idx =np.where(np.isnan(ir2))
        ir2[idx]= -1e-5

        #Process model input (IR and monthly rain) and run model
        ir_input= ir2[np.newaxis, ...]
        inp=np.stack([ir_input, xtrain2], axis=-1)
        inp=inp.astype(np.float32)
        pred0 = load_model_pred(inp,weight)

        # Apply mapping function to prediction
        pred = apply_mapping_function(pred0,satellite_quantiles,observed_quantiles)
        pred[pred<0] = 0

        # Apply Multi-Scale Guided Filtering
        radius = 15  # Adjust the radius of the filtering window as needed
        epsilon = 0.0001 # Adjust the regularization parameter as needed
        scale_factors = [1,2,4]  # Adjust the radius of the filtering window as needed
        rainfall_emphasis_factor = 1.1 # Adjust the regularization parameter as needed
        rain_threshold = 0.05 #

        guide_image = ir1.astype(np.float32)
        guide_image = cv2.normalize(guide_image, None, 0, 1, cv2.NORM_MINMAX)
    
        # Resize pred after quantile mapping from 512x1536 to 3000x9000
        interpolated_precip_data = cv2.resize(pred, (3600,1200), interpolation = cv2.INTER_LINEAR)
        interpolated_precip_data = interpolated_precip_data.astype(np.float32)
        norain_indices = np.where(interpolated_precip_data < rain_threshold)

        nan_indices = np.isnan(interpolated_precip_data)
        interpolated_precip_data[nan_indices] = 1e-3
        weighted_guide_image = interpolated_precip_data * guide_image
        weighted_guide_image[nan_indices] = 1e-3
        filtered_result = multi_scale_guided_filter(weighted_guide_image, interpolated_precip_data, radius, epsilon, scale_factors)
        filtered_result[mask] = np.nan

        # Normalize Filtered Results to Preserve Total Rainfall
        total_rain_original = np.sum(interpolated_precip_data)
        filtered_result = np.maximum(filtered_result, 1e-6)  # Avoid invalid values for power operation
        filtered_result = np.power(filtered_result, rainfall_emphasis_factor)
        filtered_result[norain_indices] = 0
        # Preventing rainfall estimates from exceeding a specified maximum threshold in Rain1hmax
        filtered_result[filtered_result > Rain1hmax] = Rain1hmax[filtered_result > Rain1hmax]

        # Rescale to Match Total Rainfall
        total_rain_after_emphasis = np.sum(filtered_result)
        if total_rain_after_emphasis > 0:
            final_scaling_factor = total_rain_original / total_rain_after_emphasis
        else:
            final_scaling_factor = 1.0
        filtered_result *= final_scaling_factor

        # Save to bin.gz file in OUTPUT directory, adjust index based on the input name
        file_path = os.path.join(OUTPUT_DIR, 'PUnet' + os.path.basename(file_ir)[5:15]+ timestep +'.bin.gz')

        # Convert output to integer format for saving to a binary file.
        # Note: When reading the file later, divide by 100 to restore the original value.
        pred_save = np.round(filtered_result * 100)

        pred_save[mask]=-9999
        pred_save = pred_save.astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin", mode='wb') as temp_file:
            # Save the array to the temporary binary file
            pred_save.tofile(temp_file, sep="", format="%<h")
            
        # Compress the temporary binary file into a gzipped binary file
        with open(temp_file.name, 'rb') as temp_bin_file, gzip.open(file_path, 'wb', compresslevel=5) as f:
            f.write(temp_bin_file.read())

        # Remove the temporary binary file
        os.remove(temp_file.name)
    
    return file_ir[5:15]

###############################################
#Main#
if __name__ == "__main__":

    start_year = int(str(startday)[:4])
    end_year = int(str(endday)[:4])
    start_month = int(str(startday)[4:6])
    end_month = int(str(endday)[4:6])

    # Generate unique (year, month) pairs to process
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue  # Skip months outside range
            year_months.append((year, f"{month:02d}"))  # Format month as '01', '02', etc.

    # Load precomputed quantile and model parameters
    observed_quantiles_all = np.load(os.path.join(PARAMETER_DIR, 'Obs_quantiles.npy'), allow_pickle=True).item()
    satellite_quantiles_all = np.load(os.path.join(PARAMETER_DIR, 'Satellite_quantiles.npy'), allow_pickle=True).item()
    Rain1hmax_all = np.load(os.path.join(PARAMETER_DIR, 'Rain1hmax01.npy'), allow_pickle=True).item()
    RAIN = np.load(os.path.join(PARAMETER_DIR, 'RAIN.npy'), allow_pickle=True).item()

    for year, month in year_months:
        print(year,month)
        # Load month-specific parameters
        weight = np.load(os.path.join(MODEL_WEIGHT_DIR, f'model_weight{month}.npy'), allow_pickle=True).item()
        xtrain2 = RAIN.get(month)
        observed_quantiles = observed_quantiles_all.get(month)
        satellite_quantiles = satellite_quantiles_all.get(month)
        Rain1hmax = Rain1hmax_all.get(month)

        # Check IR files within the date range
        all_files = sorted(glob.glob(os.path.join(IR_DATA_DIR, f'merg_{str(year)}{month}*.gz')))

        # Filter files based on startday and endday
        filenames = []
        for file in all_files:
            basename = os.path.basename(file)
            timestamp_str = basename[5:15]  # Extract YYMMDDHH
            timestamp = int(f"{timestamp_str}")  # Convert to yyyymmddhh format
            if startday <= timestamp <= endday:
                filenames.append(file)

        print(f"Processing {len(filenames)} files for {year}-{month}")

        # Process files only if they exist
        if filenames:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
                total_files = len(filenames)
                futures = {executor.submit(process_file, fn): fn for fn in filenames}
                progress_bar = tqdm(total=total_files, desc=f"Processing {year}-{month}", position=0, leave=True)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        progress_bar.update(1)
                       
                    except Exception as e:
                        print(f"Exception: {e}")

    print(f"Processing complete for files from {startday} to {endday}.")

###############################################
