import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")



x_np_train = np.genfromtxt("Data/stars_train.csv", delimiter=',', skip_header=1)
obj_ID = x_np_train[:,0]
Y = x_np_train[:,-1]
Y_np_train = Y.astype(int)
X_np_train = x_np_train[:,1:9]

x_np_test = np.genfromtxt("Data/stars_test.csv", delimiter=',', skip_header=1)
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


def submission(model,X_test = X_np_test,
               X_train = X_np_train, Y_train = Y_np_train,
               name_file = "soumission.csv"):
    
    model,scaler = train_and_eval(model, X_train, Y_train, full_train=True)
    X_test = scaler.transform(X_test)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'obj_ID': obj_ID_test , 'label': prediction})
    submission_data.to_csv(name_file, index=False)

    return submission_data


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