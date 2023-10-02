import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor

data_train = pd.read_csv("Data\wine_train.csv", index_col=0)
X = data_train.drop("target",axis=1)
Y = data_train.target

data_test = pd.read_csv("Data\wine_test.csv", index_col=0)

def train_and_eval(model, X, Y, random_state=0,full_train = False):
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
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15,random_state = random_state if random_state !=0 else None)
        # Normalisation
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled,X_test_scaled = scaler.transform(X_train),scaler.transform(X_test)
        X_train_scaled, X_test_scaled = pd.DataFrame(X_train_scaled,columns = X.columns), pd.DataFrame(X_test_scaled,columns= X.columns)

        # Entraînement du modèle
        model.fit(X_train_scaled,Y_train)
        X_result =  model.predict(X_test_scaled)
        # Evaluation : r2 score 
        score =  [r2_score(Y_test, X_result ),mean_squared_error(Y_test,X_result)**(1/2)]

        return X_result,Y_test,score

    else:
        scaler = StandardScaler()
        scaler.fit(X)
        X_used = pd.DataFrame(scaler.transform(X),columns = X.columns)
        
        # Entraînement du modèle
        model.fit(X_used,Y)
        
        return model
    pass

def multi_test(model,X,Y,n = 25,random_start=0,display_boxplot=True):
    '''
    Arguments :
        - model (instance de modèle de régression scikit-learn)
        - X : covariables
        - Y : variable à prédire
        - n : nombre de modèle à générer
        - display_boxplot : si on veut afficher le boxplot
    '''
    all_scores = [train_and_eval(model,X,Y,random_state=k)[2] for k in range(random_start,random_start+n)]
    scores_r2 = [all_scores[i][0] for i in range(len(all_scores))]
    scores_rmse = [all_scores[i][1] for i in range(len(all_scores))]
    all_scores = [scores_r2,scores_rmse]

    if display_boxplot:
        plt.boxplot(all_scores,labels=["r2_score","rmse"])
        plt.title(f"Result after training {n} {model} with different random_state")
        plt.grid()
        plt.show()
        
    return all_scores


    

def submission(model,X_test=data_test,X_train=X,Y_train=Y,name_file = "soumission.csv"):
    scaler = StandardScaler()
    scaler.fit(X_train)
    index_list = X_test.index
    model = train_and_eval(model,X_train,Y_train,full_train=True)
    X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'wine_ID': index_list, 'target': prediction})
    submission_data.to_csv(name_file, index=False)
    return submission_data