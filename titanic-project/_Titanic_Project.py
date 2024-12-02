import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class DataSet:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.features = None

    def load_df(self):
        """Load training and testing datasets."""
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.train_df.set_index('PassengerId', inplace=True)
        self.test_df.set_index('PassengerId', inplace=True)
        #self.train_df = self.train_df.drop(['Cabin'])
        #self.test_df = self.test_df.drop(['Cabin'])
        self.features = self.train_df.columns.tolist()
        print(f"Features: \n {self.features}")
        return self.train_df, self.test_df

    def preprocessing(self):
        """Preprocess data: convert categorical features, create new features, and fill missing values."""
        cat_features = ["Pclass", "Sex", "Embarked", "Cabin"]
        self.train_df = self._convert_cat(self.train_df, cat_features)
        self.test_df = self._convert_cat(self.test_df, cat_features)

        # Convert 'SibSp' and 'Parch' to integers before adding
        self.train_df['SibSp'] = self.train_df['SibSp'].astype(int)
        self.train_df['Parch'] = self.train_df['Parch'].astype(int)
        self.test_df['SibSp'] = self.test_df['SibSp'].astype(int)
        self.test_df['Parch'] = self.test_df['Parch'].astype(int)

        self.train_df['FamilyCat'] = self.train_df['SibSp'] + self.train_df['Parch'] + 1
        self.test_df['FamilyCat'] = self.test_df['SibSp'] + self.test_df['Parch'] + 1

        self.train_df['Title'] = self.train_df['Name'].apply(self.get_title)
        self.test_df['Title'] = self.test_df['Name'].apply(self.get_title)

        self.train_df['Title'] = self.train_df['Title'].apply(self.group_title)
        self.test_df['Title'] = self.test_df['Title'].apply(self.group_title)

        self.train_df['FamilyCat'] = self.train_df['FamilyCat'].apply(self.group_family_size)
        self.test_df['FamilyCat'] = self.test_df['FamilyCat'].apply(self.group_family_size)

        # Fill missing values for Age
        self.train_df['Age'] = self.train_df.groupby(['Title', 'Pclass'], observed=False)['Age'].transform(lambda x: x.fillna(x.mean()))
        self.train_df['Age'].fillna(self.train_df['Age'].mean(), inplace=True)

        self.test_df['Age'] = self.test_df.groupby(['Title', 'Pclass'], observed=False)['Age'].transform(lambda x: x.fillna(x.mean()))
        self.test_df['Age'].fillna(self.test_df['Age'].mean(), inplace=True)

        # Fill missing values for Embarked and Fare
        self.train_df['Embarked'].fillna(self.train_df['Embarked'].mode()[0], inplace=True)
        self.test_df['Embarked'].fillna(self.test_df['Embarked'].mode()[0], inplace=True)
        self.train_df['Fare'].fillna(self.train_df['Fare'].median(), inplace=True)
        self.test_df['Fare'].fillna(self.test_df['Fare'].median(), inplace=True)

        return self.train_df, self.test_df

    @staticmethod
    def group_family_size(family_size):
        """Group family sizes into categories."""
        if family_size == 1:
            return "Single"
        elif family_size <= 4:
            return "Small"
        elif family_size <= 6:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def _convert_cat(df, features):
        """Convert specific columns to categorical type."""
        for feature in features:
            df[feature] = df[feature].astype('category')
        return df

    @staticmethod
    def get_title(name):
        """Extract title from passenger's name."""
        p = re.compile(r",([\w\s]+)\.")
        return p.search(name).group(1).strip()

    @staticmethod
    def group_title(name):
        """Group titles into broader categories."""
        if name in ['Mr', 'Miss', 'Mrs', 'Master']:
            return name
        elif name == "Ms":
            return "Miss"
        else:
            return "Other"

    def feature_engineering(self):
        """Perform feature engineering."""
        num_features = ['Age', 'Fare']
        cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilyCat']

        self.train_df = self.train_df[num_features + cat_features]
        self.test_df = self.test_df[num_features + cat_features]

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ]
        )

        self.train_df = preprocessor.fit_transform(self.train_df)
        #self.test_df = preprocessor.transform(self.test_df)  # Ensure consistent processing for test data
        return self.train_df
    


class Modeling:
    def __init__(self, df_processed):
        self.X = df_processed[:, :-1]
        self.y = df_processed[:, -1]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, train_size=0.8, random_state=2024)

    def model_compare(self, X, y, metrics, cv=5):
        models = [
            LogisticRegression(random_state=2024, max_iter=1000),
            DecisionTreeClassifier(random_state=2024),
            RandomForestClassifier(random_state=2024),
            GradientBoostingClassifier(random_state=2024),
            ExtraTreesClassifier(random_state=2024),
            AdaBoostClassifier(random_state=2024),
            LinearSVC(random_state=2024, max_iter=1000),
            SVC(random_state=2024),
            KNeighborsClassifier(metric="minkowski", p=2),
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2024)
        ]
        kfold = StratifiedKFold(cv, shuffle=True, random_state=2024)
        results = []
        for model in models:
            model_name = model.__class__.__name__
            scores = cross_val_score(model, X, y, cv=kfold, scoring=metrics)
            for fold_idx, score in enumerate(scores):
                results.append((model_name, fold_idx, score))

        cv_results = pd.DataFrame(results, columns=['model_name', 'fold_id', 'accuracy_score'])

        mean = cv_results.groupby('model_name')['accuracy_score'].mean()
        std = cv_results.groupby('model_name')['accuracy_score'].std()

        baseline_results = pd.concat([mean, std], axis=1, ignore_index=True)
        baseline_results.columns = ['Mean', 'Standard Deviation']
        baseline_results.sort_values(by='Mean', ascending=False, inplace=True)
        return baseline_results
    
    def extra_trees_grid_search(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],  
                'classifier__max_depth': [None, 10, 20, 30],  
                'classifier__min_samples_split': [2, 5, 10],  
                'classifier__min_samples_leaf': [1, 2, 4],  
                'classifier__bootstrap': [True, False]  
            }

        # Xây dựng pipeline cho mô hình Extra Trees
        model = Pipeline(steps=[
            ('classifier', ExtraTreesClassifier(random_state=2024))
        ])

        # Tạo đối tượng GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(5),
            n_jobs=-1,  
            verbose=2  
        )

        # Huấn luyện mô hình với GridSearchCV
        grid_search.fit(self.X_train, self.y_train)

        
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

        
        y_pred = grid_search.predict(self.X_val)

        
        report = classification_report(self.y_val, y_pred)
        print("Classification Report:\n", report)

        return grid_search.best_estimator_,report
    
    def KNN(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'classifier__n_neighbors': [3, 5, 7, 10],  
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],  
                'classifier__p': [1, 2]
            }

        # Xây dựng pipeline với KNeighborsClassifier
        model = Pipeline(steps=[
            ('classifier', KNeighborsClassifier(n_jobs=-1)) 
        ])

        # Tạo GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(5),
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(self.X_train, self.y_train)

        y_pred = grid_search.predict(self.X_val)

        report = classification_report(self.y_val, y_pred)
        return grid_search.best_estimator_, report


    

if __name__ == "__main__":
    data_set = DataSet(
        train_path="D:\\Git\\MNT-project\\titanic-project\\train.csv",
        test_path="D:\\Git\\MNT-project\\titanic-project\\test.csv"
    )
    data_set.load_df()
    data_set.preprocessing()
    df_processed = data_set.feature_engineering()

    # Modeling and evaluation
    modeling = Modeling(df_processed)
    results = modeling.model_compare(modeling.X_train, modeling.y_train, metrics="accuracy", cv=5)
    Extra_trees,report_Ex = modeling.extra_trees_grid_search()
    KNNClassifier,report_KNN = modeling.KNN()
    with open('model_results.txt', 'w') as file:
        file.write(f"Result of Models Compare: \n {results} \n")
        file.write(f"Result of ExtraTrees Classifier:\n{report_Ex}\n")
        file.write(f"Result of KNN Classifier:\n{report_KNN}\n")
    
