from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import os.path

import torch
import torch.optim as optim
from torchvision.models import resnet50,densenet121
from torchvision.transforms import v2
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score,f1_score


from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
from src.data.pytorch_dataset import MaskingDataset
from src.models.utils import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid_epoch(model,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    lst_labels = []
    lst_preds = []
    lst_probas = []
    auc_scores = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
            lst_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)
        
        lst_labels = np.array(lst_labels)
        lst_preds = np.array(lst_preds)
        lst_probas = np.array(lst_probas)
        for i in range(lst_labels.shape[1]):
            labels = lst_labels[:,i]
            probas = lst_probas[:,i]
            auc_score=roc_auc_score(labels,probas)
            auc_scores.append(auc_score)
    return auc_scores

def main():
    #Get hyperparameters 
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    TEST = False
    models_names=["NormalDataset","NoDiscDataset_0","NoDiscBB_0","OnlyDisc_0","OnlyDiscBB_0"]
    
    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed/Train")
    testing_data = MaskingDataset(data_dir="./data/processed/Test")
    

    #Create k-fold for train/val
    stratified_group_kfold = StratifiedGroupKFold(n_splits=NB_FOLDS)
    y = np.array(training_data.img_labels["Onehot"].tolist())[:,0]

    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoDisc":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoDiscBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyDisc":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyDiscBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }

    if TEST:
        with open("./data/interim/test_results.csv", "w") as csv_file:
            csv_file.write("training_set,valid_set,class,fold,auc")

    else:
        with open("./data/interim/valid_results.csv", "w") as csv_file:
            csv_file.write("training_set,valid_set,class,fold,auc")

    for model_name in models_names:
        for param_config_name in valid_params:
            print(model_name,param_config_name)
            for i, (train_index,val_index) in enumerate(stratified_group_kfold.split(X=training_data.img_labels, y=y, groups= training_data.img_labels['PatientID'])):        
                print("\nFOLD",i)

                if TEST:
                    testing_data = MaskingDataset(data_dir="./data/processed/Test",**valid_params[param_config_name])
                    valid_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE)
                else:
                    val_data = MaskingDataset(data_dir="./data/processed/Train",**valid_params[param_config_name])
                    val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
                    val_data.img_paths = np.array(training_data.img_paths)[val_index]
                    val_data.roi_paths = np.array(training_data.roi_paths)[val_index]
                    valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

                
                
                #Define model
                weights = {
                    "name":model_name,
                    "fold":i
                }
                model = get_model(CLASSES,weights)

                val_metric = valid_epoch(model,valid_dataloader)
                if TEST:
                    with open("./data/interim/test_results.csv", "a") as csv_file:
                        for j,c in enumerate(CLASSES):
                            csv_file.write(f"\n{model_name},{param_config_name},{c},{i},{val_metric[j]}")
                            print(c,val_metric[j])    
                else:
                    with open("./data/interim/valid_results.csv", "a") as csv_file:
                        for j,c in enumerate(CLASSES):
                            csv_file.write(f"\n{model_name},{param_config_name},{c},{i},{val_metric[j]}")
                            print(c,val_metric[j])    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
