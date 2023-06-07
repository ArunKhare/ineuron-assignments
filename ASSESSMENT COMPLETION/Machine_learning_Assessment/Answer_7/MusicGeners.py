import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn import metrics
from sklearn.impute import SimpleImputer
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from numpy import unique
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
import json
import streamlit as st


class DataPreprocessingApp:
    threshold_values = np.arange(0.1, 0.9, 0.1)
    missing_values_threshold = .2
    
    def __init__(self):
        self.is_na = False
        self.model_list = [KMeans ]
        # , DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering, GaussianMixture,Birch]
        self.root_dir = os.path.join(os.getcwd(),'Answer_7')

    def get_data(self):
        file2 = Path("data.csv")
        path = Path(os.path.join(self.root_dir,file2))
        df = pd.read_csv(path)
        return df
    
    def preprocessing(self, df):
        df.drop(columns=['filename'], inplace=True)
        
        data_na = (df.isnull().sum() / len(df)) 
        data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
        
        if not data_na.empty:
            self.is_na = True
            assert self.is_na == True
        
        if (data_na >= DataPreprocessingApp.missing_values_threshold).any(): 
           rows_to_drop = df.index[df.isnull().sum(axis=1) >= DataPreprocessingApp.missing_values_threshold]
           df.drop(rows_to_drop, inplace=True) 

        y = df['label']

        cat_col = df.select_dtypes(include='O')
        num_col = df.select_dtypes(exclude='O')
        
        scaler = StandardScaler()
        num_col_scaled = scaler.fit_transform(num_col)
        num_col_scaled = pd.DataFrame(num_col_scaled, columns = scaler.get_feature_names_out())

        cat_encoded = pd.DataFrame()
        
        if df.shape[0] < 2 and not cat_col.empty:
            cat_encoded = pd.get_dummies(cat_col, drop_first=True, dummy_na=True)
        elif df.shape[0] > 2 and not cat_col.empty:
            cat_encoded = pd.get_dummies(cat_col, drop_first=True)

        if not cat_encoded.empty:
            return pd.concat([num_col_scaled, cat_encoded],axis=1), y
        
        return num_col_scaled, y
        
    @staticmethod
    def missing_value_imputer(X_train):
        imp = SimpleImputer(strategy='mean')
        imp_train = imp.fit(X_train)
        X_train = imp_train.transform(X_train)
        return X_train
    
    def train_models(self, X, y):

        training_data = X
        labels_true = y
        scoring = []
        
        map_labels = {}
        for i, label in zip(range(0,10),labels_true.unique()):
            map_labels[label] = i
        
        labels_true= labels_true.map(map_labels)
        model_list = ["KMeans", "Birch", "GaussianMixture"]

        models = {
            "Birch" : Birch(threshold=0.5, n_clusters=10),
            "KMeans" : KMeans(n_clusters=10, n_init='auto', max_iter=200, random_state=42, algorithm='lloyd'),
            "GaussianMixture" : GaussianMixture(n_components=10)
        }

        for model_key, model in models.items():

            model.fit(training_data)
            
            if model_key == model_list[0] or model_key == model_list[1]:   
           
                labels = model.labels_
                n_clusters_ = len(unique(labels))
            
            elif model_key == model_list[2]:

                # assign each data point to a cluster
                gaussian_result = model.predict(training_data)
                gaussian_clusters = unique(gaussian_result)
                labels = gaussian_result
                n_clusters_ = len(gaussian_clusters)


            model_metrics = {
                "model": model_key,
                'n_clusters': n_clusters_,
                'Homogeneity': round(metrics.homogeneity_score(labels_true, labels),3),
                'Completeness': round(metrics.completeness_score(labels_true, labels),3),
                'V-measure': round(metrics.completeness_score(labels_true, labels),3),
                'Adjusted_Rand_Index': round(metrics.adjusted_rand_score(labels_true, labels),3),
                'Adjusted_Mutual_Information': round(metrics.adjusted_rand_score(labels_true, labels),3),
                'Silhouette Coefficient': round(metrics.silhouette_score(training_data, labels, metric="sqeuclidean"),3),
                'pair_confusion_matrix': pair_confusion_matrix(labels_true,labels).tolist()
            }
            
            scoring.append(model_metrics)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
            print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
            print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
            print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
            print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(training_data, labels, metric="sqeuclidean") )
            print("pair_confusion_matrix:",  pair_confusion_matrix(labels_true,labels))

        # Store the scoring list as JSON
        path = os.path.join(self.root_dir,"scoring.json")
        with open(path, 'w') as f:
           json.dump(scoring, f)

        return model_metrics, model
    
    def save_model(self, best_estimator):
        self.model_path = Path(os.path.join(self.root_dir,'music_geners.pkl'))
        joblib.dump(best_estimator, self.model_path)
    

    
    def display_results(self):
        # Load the scoring results from JSON
        scoring_path = os.path.join(self.root_dir, "scoring.json")
        with open(scoring_path, 'r') as f:
            scoring = json.load(f)

        # Display the models and model scores
        st.header("Model Scores")
        for model_metrics in scoring:
            st.subheader(f"Model: {model_metrics['model']}")
            st.text(f"Number of Clusters: {model_metrics['n_clusters']}")
            st.text(f"Homogeneity: {model_metrics['Homogeneity']}")
            st.text(f"Completeness: {model_metrics['Completeness']}")
            st.text(f"V-measure: {model_metrics['V-measure']}")
            st.text(f"Adjusted Rand Index: {model_metrics['Adjusted_Rand_Index']}")
            st.text(f"Adjusted Mutual Information: {model_metrics['Adjusted_Mutual_Information']}")
            st.text(f"Silhouette Coefficient: {model_metrics['Silhouette Coefficient']}")
            st.text(f"Pair Confusion Matrix: {model_metrics['pair_confusion_matrix']}")

        # Display some data
        st.header("Data Sample")
        df = self.get_data()
        st.dataframe(df.head())

    def initiate_model_training(self):
        df = self.get_data()
        X, y = self.preprocessing(df=df)
        results, best_estimator = self.train_models(X, y)
        self.save_model(best_estimator=best_estimator)
        self.display_results()

    def __del__(self):
        print(f"{'>>'*20} Training complete {'<<'*20}")


if __name__ == "__main__":
    app = DataPreprocessingApp()
    app.initiate_model_training()