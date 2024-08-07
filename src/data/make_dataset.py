# -*- coding: utf-8 -*-
from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import ast
import glob
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
import torch
from src.models.lung_segmentation.model.Unet import Unet
import tensorflow as tf


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('classes', default=None,type=str)
def main(input_filepath, output_filepath, classes):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    os.makedirs(os.path.dirname(f"./{output_filepath}/Train/images/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{output_filepath}/Train/rois/"), exist_ok=True)

    os.makedirs(os.path.dirname(f"./{output_filepath}/Test/images/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{output_filepath}/Test/rois/"), exist_ok=True)

    assert os.path.isdir(f"./{output_filepath}/Train/images")
    assert os.path.isdir(f"./{output_filepath}/Train/rois")

    assert os.path.isdir(f"./{output_filepath}/Test/images")
    assert os.path.isdir(f"./{output_filepath}/Test/rois")

    logger.info(f'Processing labels and store the result in ./{output_filepath}/processed_labels.csv')
    filter_and_process_labels(input_filepath,output_filepath,classes)
    assert os.path.exists(f"{output_filepath}/Train/processed_labels.csv")
    assert os.path.exists(f"{output_filepath}/Test/processed_labels.csv")

    if os.listdir(f"./{output_filepath}/Train/images") == []:
        logger.info(f'Creating image dataset in ./{output_filepath}/images')
        create_images(input_filepath,output_filepath)
        assert os.listdir(f"./{output_filepath}/Train/images")
    else:
        logger.info(f'./{output_filepath}/images not empty, skipping creation of images')
    
    if os.listdir(f"./{output_filepath}/Train/rois") == []:
        logger.info(f'Creating ROIs of images in ./{output_filepath}/rois')
        create_rois(input_filepath,output_filepath)
        assert os.listdir(f"./{output_filepath}/Train/rois")
    else:
        logger.info(f'./{output_filepath}/rois not empty, skipping creation of rois')

    logger.info(f'Dataset is ready to be used!')




def filter_and_process_labels(input_filepath,output_filepath,classes):
    for split in ["Train","Test"]:
        #Get base labels
        base_path_labels = f"./data/raw/{split}/6.0_Glaucoma_Decision"

        df_bosch = pd.read_csv(f'{base_path_labels}/Glaucoma_Decision_Comparison_Bosch_majority.csv')
        #Rename label columns to fit the others
        df_bosch.rename({"Majority Decision":"Glaucoma Decision"},axis=1,inplace=True)

        df_forus = pd.read_csv(f'{base_path_labels}/Glaucoma_Decision_Comparison_Forus_majority.csv')
        df_remidio = pd.read_csv(f'{base_path_labels}/Glaucoma_Decision_Comparison_Remidio_majority.csv')

        #Create ImageID as the real file name and extension
        df_bosch["ImageID"] = df_bosch["Images"].apply(lambda x:x.split("-")[0].replace("jpg","JPG"))
        df_forus["ImageID"] = df_forus["Images"].apply(lambda x:x.split("-")[0].replace("jpg","png"))
        df_remidio["ImageID"] = df_remidio["Images"].apply(lambda x:x.split("-")[0].replace("tif","JPG"))


        #Add device before concat all labels
        df_bosch["Camera"] = "Bosch"
        df_forus["Camera"] = "Forus"
        df_remidio["Camera"] = "Remidio"
         
        #Concat all labels together
        df_all_source = pd.concat([df_bosch,df_forus,df_remidio])


        #Remove labels for images not present in the data
        base_path_images = f"./data/raw/{split}/1.0_Original_Fundus_Images"
        images_name = [p.split("/")[-1] for p in glob.glob(f"{base_path_images}/*/*")]
        missing_images = ~(df_all_source["ImageID"].isin(images_name))
        df_all_source_present_images = df_all_source[~missing_images]

        #Change the label to binary value instead of "NORMAL" and "GLAUCOMA SUSPECT"
        df_all_source_present_images["Onehot"] = df_all_source_present_images["Glaucoma Decision"].apply(lambda x: [1 if x == "GLAUCOMA SUSPECT" else 0])

        #Add PatientID to avoid images from a same patient in different split, only useful for Bosch images starting with PXX_ where XX could be the PatientID
        df_all_source_present_images["PatientID"] = df_all_source_present_images["ImageID"].apply(lambda x: x.split("_")[0] if x.startswith("P") else x[:-4])

        #Only keep necessary columns
        df_processed = df_all_source_present_images[["ImageID","PatientID","Camera","Onehot"]]
        df_processed = df_processed.reset_index(drop=True)
        df_processed.to_csv(f"{output_filepath}/{split}/processed_labels.csv",sep=",",index=False)

def create_images(input_filepath,output_filepath):
    for split in ["Train","Test"]:
        #Load labels
        labels = pd.read_csv(f"{output_filepath}/{split}/processed_labels.csv")
        labels["Onehot"] = labels["Onehot"].apply(lambda x: ast.literal_eval(x))
    
        #Get images present at input_filepath
        images_path = glob.glob(f"./{input_filepath}/{split}/1.0_Original_Fundus_Images/**/*.*",recursive=True)
        image_names = [path.split('/')[-1] for path in images_path]
        for idx,i_name in enumerate(tqdm(image_names)):
            #Resize the image and save it in the processed folder
            if i_name in labels["ImageID"].unique():
                img = io.imread(images_path[idx])
                max_value = np.max(img) 
                img = tf.image.resize_with_pad(img, 512, 512)
                img = img/max_value
                tf.keras.utils.save_img(f"./{output_filepath}/{split}/images/{i_name}", img, scale=True, data_format="channels_last")        

def create_rois(input_filepath,output_filepath):
        for split in ["Train","Test"]:
            print("yo")
            #Load labels
            labels = pd.read_csv(f"{output_filepath}/{split}/processed_labels.csv")
            labels["Onehot"] = labels["Onehot"].apply(lambda x: ast.literal_eval(x))
        
            #Get images present at input_filepath
            masks_path = glob.glob(f"./{input_filepath}/{split}/4.0_OD_CO_Fusion_Images/**/STAPLE/*.*",recursive=True)
            masks_names = [path.split('/')[-1] for path in masks_path]
            print(masks_path)
            for idx,m_name in enumerate(tqdm(masks_names)):
                #Resize the image and save it in the processed folder
                if m_name in labels["ImageID"].unique():
                    mask = np.expand_dims(io.imread(masks_path[idx]),-1)
                    mask_resize = tf.image.resize_with_pad(mask, 512, 512)
                    mask_resize = mask_resize > 0
                    tf.keras.utils.save_img(f"./{output_filepath}/{split}/rois/{m_name}", mask_resize, scale=True, data_format="channels_last") 

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
