import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path


import torch
from torchvision.models import resnet50,densenet121
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score,f1_score

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
from src.data.pytorch_dataset import MaskingDataset

import shap
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from src.models.utils import get_model,make_single_pred
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_auc_per_label(path_to_results="./data/interim/valid_results.csv",show=False):
    models_valid_results = pd.read_csv(path_to_results)
    mean_auc_per_class = models_valid_results.groupby(["class","training_set","valid_set"])["auc"].mean()
    for class_label in mean_auc_per_class.index.get_level_values('class').unique():
        result_class = mean_auc_per_class[mean_auc_per_class.index.get_level_values('class').isin([class_label])].droplevel(0)
        result_class = result_class.reset_index().pivot(columns='valid_set',index='training_set',values='auc')
        result_class = result_class[["NoDiscBB","NoDisc","Normal","OnlyDiscBB","OnlyDisc"]]

        plt.figure(figsize=(9,7))
        plt.title(f"Mean AUC for {class_label} across different masking strategies",size=20)
        heatmap = sns.heatmap(result_class, annot=True,cmap="RdYlGn",annot_kws={"size": 15},xticklabels=["NoDiscBB","NoDisc","Full","OnlyDiscBB","OnlyDisc"],yticklabels=["NoDiscBB","NoDisc","Full","OnlyDiscBB","OnlyDisc"])
        heatmap.yaxis.set_tick_params(labelsize = 15)
        heatmap.xaxis.set_tick_params(labelsize = 15)
        plt.xlabel('Validation set masking', fontsize=13)
        plt.ylabel('Training set masking', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"./reports/figures/mean_auc_{class_label}.png",format='png')
        if show:
            plt.show()

def generate_explainability_map():
    #Get hyperparameters 
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    model_name="NormalDataset"

    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed")
    testing_data = MaskingDataset(data_dir="./data/processed")

    #Split the dataset into training/testing splits
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 1907)
    train_eval_split = splitter.split(training_data.img_labels, groups=training_data.img_labels['PatientID'])
    train_idx, test_idx = next(train_eval_split)
    training_data.img_labels = training_data.img_labels.iloc[train_idx].reset_index(drop=True)
    training_data.img_paths = np.array(training_data.img_paths)[train_idx]
    training_data.roi_paths = np.array(training_data.roi_paths)[train_idx]

    testing_data.img_labels = testing_data.img_labels.iloc[test_idx].reset_index(drop=True)
    testing_data.img_paths = np.array(testing_data.img_paths)[test_idx]
    testing_data.roi_paths = np.array(testing_data.roi_paths)[test_idx]
    

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        # "NoLung":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        # "NoLungBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        # "OnlyLung":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        # "OnlyLungBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }

    for param_config_name in valid_params:
        print(model_name,param_config_name)
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):        
            print("\nFOLD",i)
            val_data = MaskingDataset(data_dir="./data/processed",**valid_params[param_config_name])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
            
            
            #Define model, loss and optimizer
            model = densenet121(weights='DEFAULT')#Weights pretrained on imagenet_1k
            
            # Freeze every layer except last denseblock and classifier
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features.denseblock4.denselayer16.parameters():
                param.requires_grad = True
           
            kernel_count = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
             torch.nn.Flatten(),
             torch.nn.Linear(kernel_count, len(CLASSES))
            )
            
            for module in model.modules():
                if isinstance(module, torch.nn.ReLU):
                    module.inplace = False
                    
            try:
                model.load_state_dict(torch.load(f"./models/{model_name}/{model_name}_Fold{i}.pt"))
                model.to(DEVICE)
            except FileNotFoundError as e:
                print("No model saved for fold",i)
                continue

            images, _ = next(iter(valid_dataloader))
            images = images.to(DEVICE)
            background = images[:1]
            test_images= images[1:]
            e = shap.DeepExplainer(model, images)
            shap_values = e.shap_values(test_images)


def get_embedding(model_name,valid_params):
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    
    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed/Train")
    testing_data = MaskingDataset(data_dir="./data/processed/Test")

    y = np.array(training_data.img_labels["Onehot"].tolist())[:,0]

    #Create k-fold for train/val
    stratified_group_kfold = StratifiedGroupKFold(n_splits=NB_FOLDS)
    
    
    models_flatten_output = {
        masking_param:[] for masking_param in valid_params
    }
    
    for masking_param in valid_params:
        print(f"\n{masking_param}")
        for i, (train_index,val_index) in enumerate(stratified_group_kfold.split(X=training_data.img_labels, y=y, groups= training_data.img_labels['PatientID'])):
            val_data = MaskingDataset(data_dir="./data/processed/Train",**valid_params[masking_param])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
            
            #Define model
            weights = {
                "name":model_name,
                "fold":i
            }
            
            return_nodes = {
                "classifier.0": "flatten"
            }
            model = get_model(CLASSES,weights,return_nodes=return_nodes)
            model.eval()
            
            with torch.no_grad():
                labels_dataset = []
                for i, data in enumerate(valid_dataloader, 0):
                    inputs, labels = data
                    inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = model(inputs)
                    models_flatten_output[masking_param].extend(outputs["flatten"].detach().cpu().tolist())
                    labels_dataset += labels
            models_flatten_output[masking_param] = np.array(models_flatten_output[masking_param])
            break
    return models_flatten_output, labels_dataset


def get_cosine():
    model_name="NormalDataset"
    
    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoDisc":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoDiscBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyDisc":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyDiscBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }
    CLASSES = os.environ.get("CLASSES").split(",")
    
    models_flatten_output,labels_dataset = get_embedding(model_name,valid_params)

    no_disc_similarities = []
    only_disc_similarities = []
    no_discbb_similarities = []
    only_discbb_similarities = []
    for j in range(len(models_flatten_output["Normal"])):
        normal = models_flatten_output["Normal"][j]
        nodisc = models_flatten_output["NoDisc"][j]
        nodiscbb = models_flatten_output["NoDiscBB"][j]
        onlydisc = models_flatten_output["OnlyDisc"][j]
        onlydiscbb = models_flatten_output["OnlyDiscBB"][j]
        
        no_disc_similarities.append(1- cosine(normal,nodisc))
        no_discbb_similarities.append(1- cosine(normal,nodiscbb))
        only_disc_similarities.append(1- cosine(normal,onlydisc))
        only_discbb_similarities.append(1- cosine(normal,onlydiscbb))
    print(f"all,{np.mean(no_disc_similarities)}+/-{np.std(no_disc_similarities)},\
            {np.mean(no_discbb_similarities)}+/-{np.std(no_discbb_similarities)},\
            {np.mean(only_disc_similarities)}+/-{np.std(only_disc_similarities)},\
            {np.mean(only_discbb_similarities)}+/-{np.std(only_discbb_similarities)}")

    #Per class
    for i,c in enumerate(CLASSES):
        no_disc_similarities = []
        only_disc_similarities = []
        no_discbb_similarities = []
        only_discbb_similarities = []
        class_indices = [j for j, l in enumerate(labels_dataset) if l[i] == 1]
        for j in class_indices:
            normal = models_flatten_output["Normal"][j]
            nodisc = models_flatten_output["NoDisc"][j]
            nodiscbb = models_flatten_output["NoDiscBB"][j]
            onlydisc = models_flatten_output["OnlyDisc"][j]
            onlydiscbb = models_flatten_output["OnlyDiscBB"][j]
            
            no_disc_similarities.append(1- cosine(normal,nodisc))
            no_discbb_similarities.append(1- cosine(normal,nodiscbb))
            only_disc_similarities.append(1- cosine(normal,onlydisc))
            only_discbb_similarities.append(1- cosine(normal,onlydiscbb))
        print(f"{c},{np.mean(no_disc_similarities)}+/-{np.std(no_disc_similarities)},\
            {np.mean(no_discbb_similarities)}+/-{np.std(no_discbb_similarities)},\
            {np.mean(only_disc_similarities)}+/-{np.std(only_disc_similarities)},\
            {np.mean(only_discbb_similarities)}+/-{np.std(only_discbb_similarities)}")


def plot_tsne(model_name="NormalDataset",show=False):
    """
    Create t-SNE plot of the embeddings of images with different masking produced by the last layer before classification head
    @param:
        model_name: str, default "NormalDataset". Name of the model weights to use
        show: bool, default False. Save and show the plot if True, only save it otherwise
    """
    model_name="NormalDataset"
      
    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoDisc":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoDiscBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyDisc":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyDiscBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }
    
    
    models_embeddings,labels_dataset = get_embedding(model_name,valid_params)
    
    #We convert the dict from get_embedding to regroup them all (not group by masking anymore) in an array to perform the t-SNE
    models_flatten_output = []
    
    #This array will keep the info on the masking used to produce this embedding, useful for visualisation later
    labels_masking = []
    for masking in models_embeddings:
        models_flatten_output.extend(models_embeddings[masking])
        labels_masking += [masking] * len(models_embeddings[masking])    
    models_flatten_output = np.array(models_flatten_output)
    
    #Taken from https://learnopencv.com/t-sne-for-feature-visualization/
    tsne = TSNE(n_components=2,perplexity=10).fit_transform(np.array(models_flatten_output))
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
     
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
     
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range
     
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
     
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)


    #Plot divided by masking strategy
    plt.figure()
    for masking_param in valid_params:
        indices = [j for j, l in enumerate(labels_masking) if l == masking_param]
        # extract the coordinates of the points of the current masking
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx,current_ty,label=masking_param,alpha=0.5)
    plt.title(f"t-SNE of the embeddings before classication head of images with different masking strategy",size=15)
    plt.legend(bbox_to_anchor=(1,1),fontsize=15)
    plt.savefig(f"./reports/figures/tsne.png",format='png',bbox_inches='tight')
    if show:
        plt.show()


def get_mean_std_proba(model,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            if sum(labels) == 0:
                continue
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
        
        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)
    return np.mean(lst_probas),np.std(lst_probas)


def dilation_impact_auc(model_name,masking_param,dilation_factors,class_to_dilate=0):
    """
    Apply validation set and compute AUC for images of the validation sets with increasing dilation factor
    @param:
        -model_name: name of the weights to load
        -masking_param: dict with the masking parameters {"inverse_roi":False,"bounding_box":False}
        -dilation_factors: list of the dilation factors (masking spread) to apply
        -class to dilate: Which class should be dilated 0 for healthy and 1 for glaucoma
    @return:
        -The list of AUCs for each dilation factors
    """
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = 1
    CLASSES = os.environ.get("CLASSES").split(",")
    
    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed/Train")
    
    y = np.array(training_data.img_labels["Onehot"].tolist())[:,0]

    #Create k-fold for train/val
    stratified_group_kfold = StratifiedGroupKFold(n_splits=NB_FOLDS)
    lst_auc = [[] for i in range(len(dilation_factors))]

    for k,dilation_factor in enumerate(dilation_factors):
        masking_param["masking_spread"]=dilation_factor
        for i, (train_index,val_index) in enumerate(stratified_group_kfold.split(X=training_data.img_labels, y=y, groups= training_data.img_labels['PatientID'])):
            
            val_data = MaskingDataset(data_dir="./data/processed/Train",**masking_param)
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
            model.eval()
        
            lst_labels = []
            lst_probas = []
            auc_scores = []
            valid_dataloader_normal = copy.deepcopy(valid_dataloader)
            valid_dataloader_normal.dataset.masking_spread = 0

            with torch.no_grad():
                for i,data in enumerate(zip(valid_dataloader,valid_dataloader_normal)):
                    if sum(data[0][1]) == class_to_dilate:
                        inputs,labels = data[0]
                    else:
                        inputs,labels = data[1]
                    
                    inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = model(inputs)
                    output_sigmoid = sigmoid(outputs)
                    lst_labels.extend(labels.cpu().detach().numpy())
                    lst_probas.extend(output_sigmoid.cpu().detach().numpy())
                
                lst_labels = np.array(lst_labels)
                lst_probas = np.array(lst_probas)
                for i in range(lst_labels.shape[1]):
                    labels = lst_labels[:,i]
                    probas = lst_probas[:,i]
                    auc_score=roc_auc_score(labels,probas)
                    auc_scores.append(auc_score)
                
                auc = auc_scores[0]
                lst_auc[k].append(auc)
    return lst_auc

def plot_impact_auc(model_name,masking_param,savefile_name,show=False):
    """
    Plot the evolution of AUC while dilating the mask for model trained with No Disc and Only Disc images
    @param:
        -model_name: name of the weights to load
        -masking_param: dict with the masking parameters {"inverse_roi":False,"bounding_box":False}
        -savefile_name: Name of the file used to save the plot
        -show: bool, default False. Save and show the plot if True, only save it otherwise
    """
    dilation_factors = [0,5,10,25,50,100,150,200,300,400,500]

    lst_auc_healthy_dilation = dilation_impact_auc(model_name,masking_param,dilation_factors,0)
    lst_auc_glaucoma_dilation = dilation_impact_auc(model_name,masking_param,dilation_factors,1)
    
    plt.figure()
    mean_auc_healthy = np.array([np.mean(lst_auc_healthy_dilation[k]) for k in range(len(lst_auc_healthy_dilation))])
    std_auc_healthy = np.array([np.std(lst_auc_healthy_dilation[k]) for k in range(len(lst_auc_healthy_dilation))])
    plt.plot(dilation_factors,mean_auc_healthy,marker="o",label="healthy images",color="tab:blue")
    plt.fill_between(dilation_factors, mean_auc_healthy-std_auc_healthy, mean_auc_healthy+std_auc_healthy,alpha=0.2,color="tab:blue")

    mean_auc_glaucoma = np.array([np.mean(lst_auc_glaucoma_dilation[k]) for k in range(len(lst_auc_glaucoma_dilation))])
    std_auc_glaucoma = np.array([np.std(lst_auc_glaucoma_dilation[k]) for k in range(len(lst_auc_glaucoma_dilation))])
    plt.plot(dilation_factors,mean_auc_glaucoma,marker="o",label="glaucoma images",color="tab:orange")
    plt.fill_between(dilation_factors, mean_auc_glaucoma-std_auc_glaucoma, mean_auc_glaucoma+std_auc_glaucoma,alpha=0.2,color="tab:orange")


    plt.title("Evolution of AUC while expanding mask's size")
    plt.xlabel("Dilation factor")
    plt.ylabel("Mean AUC across 5-fold")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"./reports/figures/{savefile_name}.png")
    savefile_name
    if show:
        plt.show()


def main():
    print("GENERATING AUC MATRICES")
    generate_auc_per_label("./data/interim/valid_results.csv")
    
    print("\nCOMPUTING COSINE SIMILARITIES")
    get_cosine()
    
    print("\nCREATING t-SNE PLOT")
    plot_tsne()
    
    print("\n STUDY OF DILATION IMPACT ON NO DISC")
    masking_param = {"inverse_roi":False,"bounding_box":False}
    plot_impact_auc("NoDiscDataset_0",masking_param,"mean_auc_dilation_nodisc",True)
    
    print("\n STUDY OF DILATION IMPACT ON ONLY DISC")
    masking_param = {"inverse_roi":True,"bounding_box":False}
    plot_impact_auc("OnlyDisc_0",masking_param,"mean_auc_dilation_onlydisc",True)

    # print("\nGENERATING EXPLAINABILITY MAPS")
    # generate_explainability_map()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
