import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.pytorch_dataset import MaskingDataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
import numpy as np

def main():
    NB_FOLDS = 5

    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed/Train")
    testing_data = MaskingDataset(data_dir="./data/processed/Test")

    y = np.array(training_data.img_labels["Onehot"].tolist())[:,0]
    #Create k-fold for train/val
    stratified_group_kfold = StratifiedGroupKFold(n_splits=NB_FOLDS)

    camera_id = {
        "Bosch":0,
        "Forus":1,
        "Remidio":2
    }
    with open("./data/interim/test_results_tabular.csv", "a") as csv_file:
        csv_file.write(f"model_name,fold,auc")
    for i, (train_index,val_index) in enumerate(stratified_group_kfold.split(X=training_data.img_labels, y=y, groups= training_data.img_labels['PatientID'])):
        print(f"Start FOLD {i}:")
        train_data = MaskingDataset(data_dir="./data/processed/Train")
        train_data.img_labels = training_data.img_labels.iloc[train_index].reset_index(drop=True)
        train_data.img_paths = np.array(training_data.img_paths)[train_index]
        train_data.roi_paths = np.array(training_data.roi_paths)[train_index]
        
        val_data = MaskingDataset(data_dir="./data/processed/Train")
        val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
        val_data.img_paths = np.array(training_data.img_paths)[val_index]
        val_data.roi_paths = np.array(training_data.roi_paths)[val_index]        
        
        test_data = MaskingDataset(data_dir="./data/processed/Test")


        X_train= np.array([camera_id[c] for c in train_data.img_labels["Camera"].tolist()]).reshape(-1, 1)
        y_train = np.array([l[0] for l in train_data.img_labels["Onehot"].tolist()])
        
        X_val= np.array([camera_id[c] for c in val_data.img_labels["Camera"].tolist()]).reshape(-1, 1)
        y_val = np.array([l[0] for l in val_data.img_labels["Onehot"].tolist()])

        X_test= np.array([camera_id[c] for c in test_data.img_labels["Camera"].tolist()]).reshape(-1, 1)
        y_test = np.array([l[0] for l in test_data.img_labels["Onehot"].tolist()])

        clf = LogisticRegression(random_state=1907).fit(X_train, y_train)
        lst_probas = [[] for l in testing_data.img_labels["Onehot"].tolist()]

        #Predict on the test set
        probas = clf.predict_proba(X_test)[:, 1]
        for k,p in enumerate(probas):
            lst_probas[k].append(p)
        auc = roc_auc_score(y_test, probas)

        #Save the probas and the auc
        with open(f"./data/interim/test_probas_Tabular_Fold{i}.csv", "a") as csv_file:
            csv_file.write(f"labels,probas")
            for label,proba in zip(testing_data.img_labels["Onehot"].tolist(),lst_probas):
                csv_file.write(f"\n{np.array(label)},{np.array(proba)}")
       
        with open("./data/interim/test_results_tabular.csv", "a") as csv_file:
            csv_file.write(f"\nTabular,{i},{auc}")
            print(auc)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
