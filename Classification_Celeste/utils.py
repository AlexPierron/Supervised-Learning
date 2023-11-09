import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from tqdm import tqdm
import copy

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

random_seed = 42 #Manage the randomness inside the models

nb_cpu = os.cpu_count()
data_train = pd.read_csv("Data\stars_train_new.csv", index_col=0)
X = data_train.drop("label",axis=1)
Y = data_train.label

data_test = pd.read_csv("Data\stars_test_new.csv", index_col=0)

x_np_train = np.genfromtxt("Data/stars_train_new.csv", delimiter=',', skip_header=1)
obj_ID = x_np_train[:,0]
Y = x_np_train[:,-1]
Y_np_train = Y.astype(int)
X_np_train = x_np_train[:,1:9]

x_np_test = np.genfromtxt("Data/stars_test_new.csv", delimiter=',', skip_header=1)
obj_ID_test = x_np_test[:,0]
X_np_test = x_np_test[:,1:]


def train_and_eval(model, X, Y, random_state=0,full_train = False,test_size=0.15):
    '''
    Arguments :
        - model (instance de modèle de régression scikit-learn)
        - X : covariables
        - Y : variable à prédire
        - random_state : seed pour l'initialisation du générateur aléatoire
        - full_train : si on veut faire un entrainement avec tout le jeu de données ou pas.
    '''

    if not full_train:
        # Train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state = random_state if random_state !=0 else None)
        # Normalisation
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled,X_test_scaled = scaler.transform(X_train),scaler.transform(X_test)

        # Entraînement du modèle
        model.fit(X_train_scaled,Y_train)
        X_result =  model.predict(X_test_scaled)
        # Evaluation : r2 score 
        score =  [r2_score(Y_test, X_result ),f1_score(Y_test,X_result,average="weighted")]

        return X_result,Y_test,score

    else:
        scaler = StandardScaler()
        scaler.fit(X)
        X_used = scaler.transform(X)
        
        # Entraînement du modèle
        model.fit(X_used,Y)
        
        return model,scaler
    pass


"""
def submission(model,X_test = X_np_test,
               X_train = X_np_train, Y_train = Y_np_train,
               name_file = "soumission.csv"):
    
    model,scaler = train_and_eval(model, X_train, Y_train, full_train=True)
    X_test = scaler.transform(X_test)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'obj_ID': obj_ID_test , 'label': prediction})
    submission_data.to_csv(name_file, index=False)

    return submission_data
"""

def multi_test(model,X = X_np_train,
               Y = Y_np_train,n = 25, test_size = 0.15,
               random_start= 0,display_boxplot=True):
    '''
    Arguments :
        - model (instance de modèle de régression scikit-learn)
        - X : covariables
        - Y : variable à prédire
        - n : nombre de modèle à générer
        - display_boxplot : si on veut afficher le boxplot
    '''
    all_scores = [train_and_eval(model,X,Y,random_state=k,test_size=test_size)[2] for k in range(random_start,random_start+n)]
    scores_r2 = [all_scores[i][0] for i in range(len(all_scores))]
    scores_f1 = [all_scores[i][1] for i in range(len(all_scores))]
    all_scores = [scores_r2,scores_f1]

    if display_boxplot:
        plt.boxplot(all_scores,labels=["r2_score","f1_score"])
        plt.title(f"Result after training {n} {model} with different random_state")
        plt.grid()
        plt.show()
        
    return all_scores

def grid_search(model, X, Y, hyperparameters, n=25, random_start=0, display_boxplot=False,
                saving_path=None, scikit_model=True):
    """
    Perform grid search with cross-validation to find the best hyperparameters for a regression model.

    Args:
        model: A regression model object, either scikit-learn compatible or custom.
        X: Features (covariates) for regression.
        Y: Target variable to predict.
        hyperparameters: Dictionary of hyperparameters and their possible values for grid search.
        n: Number of estimators.
        random_start: Random seed for initialization.
        display_boxplot: Flag to display a box plot of the evaluation results.
        saving_path: Path to save the best model, or None if not saving.
        scikit_model: True if using a scikit-learn compatible model, False if it is a PyTorch model.

    Returns:
        best_model: The best model with the optimal hyperparameters.
        best_params: The hyperparameters that resulted in the best performance.
        best_score: The best performance score (mean r2_score).
        all_results: A list of results for all hyperparameter combinations.
    """
    best_score = None
    best_params = None
    best_model = None
    all_results = []  

    if scikit_model:
        for params in tqdm(ParameterGrid(hyperparameters), desc="Grid Search Progress"):
            current_model = model.set_params(**params)

            scores = multi_test(current_model, X, Y, n=n, random_start=random_start, display_boxplot=False)
            current_f1_score = np.array(scores[1])

            if best_score is None or np.mean(current_f1_score) > best_score:
                best_score = np.mean(current_f1_score)
                best_params = params
                best_model = copy.copy(current_model)
                best_model_scores = scores

            all_results.append({
                "params": params,
                "scores": scores
            })

    else:
        for params in tqdm(ParameterGrid(hyperparameters), desc="Grid Search Progress"):
            current_model = model(**params)

            scores = multi_test(current_model, X, Y, n=n, random_start=random_start, display_boxplot=False)
            current_f1_score = np.array(scores[0])

            if best_score is None or np.mean(current_f1_score) > best_score:
                best_score = np.mean(current_f1_score)
                best_params = params
                best_model = current_model
                best_model_scores = scores

            all_results.append({
                "params": params,
                "scores": scores
            })
    print(current_model)
    if display_boxplot:
        plt.boxplot(best_model_scores, labels=["r2_score", "f1_score"])
        plt.title(f"Best Model: {best_model} with Hyperparameters: {best_params}")
        plt.grid()
        plt.show()

    print("Best hyperparameters:", best_params)
    print("Best score (f1_score):", best_score)

    if saving_path is not None:
        output_folder = os.path.dirname(saving_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        joblib.dump(best_model, saving_path)

    return best_model, best_params, best_score, all_results


def submission(model,X_test=data_test,X_train=X,Y_train=Y,name_file = "Submissions/soumission.csv",pretrained=False):
    """
    Generate a submission file using the desired model.

    Args:
        model: A trained regression model.
        X_test: Test dataset features (default: data_test).
        X_train: Training dataset features.
        Y_train: Training dataset target variable.
        name_file: Name of the output submission file (default: "submission.csv").
        pretrained: Whether or not the model is already trained

    Returns:
        submission_data: A DataFrame containing the wine_ID and predicted target.
    """

    scaler = StandardScaler()
    scaler.fit(X_train)
    index_list = X_test.index
    if not pretrained:
        model = train_and_eval(model,X_train,Y_train,full_train=True)[0]
    X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'obj_ID': index_list, 'label': prediction})

    output_folder = os.path.dirname(name_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    submission_data.to_csv(name_file, index=False)
    return submission_data


"""
def submission(model,X_test=x_np_test,X_train=X_np_train,Y_train=Y_np_train,name_file = "soumission.csv"):
    """
"""
    Generate a submission file using a trained regression model.

    Args:
        model: A trained regression model.
        X_test: Test dataset features (default: data_test).
        X_train: Training dataset features.
        Y_train: Training dataset target variable.
        name_file: Name of the output submission file (default: "submission.csv").

    Returns:
        submission_data: A DataFrame containing the wine_ID and predicted target.

    """
"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    index_list = X_test[:,0]
    X_test = x_np_test[:,1:]
    model = train_and_eval(model,X_train,Y_train,full_train=True)
    X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'wine_ID':index_list , 'target': prediction})

    output_folder = os.path.dirname(name_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    submission_data.to_csv(name_file, index=False)
    return submission_data
"""

def save_model(model,path_to_save="Archives_Model/Default_best_model.pkl"):
    output_folder = os.path.dirname(path_to_save)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    joblib.dump(model, path_to_save)