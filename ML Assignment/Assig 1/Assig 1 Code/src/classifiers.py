import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import CustomMetrics


# This file contains the function definition for all the four models used for training and testing the data
# We have used the grid search method for model tuning and used K fold cross validation for model training


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class All_Classifiers:
    """
    Complete pipeline for training and tuning 4 classifiers with grid search
    : Decision Trees, Random Forest, KNN, SVM
    """
    
    def __init__(self, train_file, test_file=None):
        """Initialize the pipeline"""
        self.random_state = RANDOM_SEED
        self.best_models = {}
        self.results = {}
        
        # Load training data
        self.df_full = pd.read_csv(train_file)
        X_full = self.df_full.iloc[:, :-1].values
        y_full = self.df_full.iloc[:, -1].values
        

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=self.random_state, stratify=y_full
        )

        self.X_train_poly = None
        self.X_val_poly = None
        self.X_test_poly = None
        self.poly_transformer = None

        # Load test data if provided
        if test_file:
            self.df_test = pd.read_csv(test_file)
            self.X_test = self.df_test.iloc[:, :].values
            self.y_test = None  
        else:
            self.X_test = None
        
        print(f"\n")
        print("CLASSIFIER TRAINING PIPELINE")
        print(f"\nPrimary Training data shape: {self.X_train.shape} (Used for tuning)")
        print(f"Validation data shape: {self.X_val.shape} (Used for final evaluation)")
        print(f"Classes: {np.unique(self.y_train)}")
        print(f"Class distribution: {dict(pd.Series(self.y_train).value_counts().sort_index())}")
        
        # Feature scaling (For SVM and KNN)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\nFeatures scaled: StandardScaler applied")

    
    def tune_decision_tree(self):
        """Tune Decision Tree Classifier with Grid Search"""
        print("\n")
        print("1. DECISION TREE CLASSIFIER - GRID SEARCH")
        
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        dt = DecisionTreeClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTuning parameters:")
        print(f"  Criterion: {param_grid['criterion']}")
        print(f"  Max Depth: {param_grid['max_depth']}")
        print(f"  Min Samples Split: {param_grid['min_samples_split']}")
        print(f"  Min Samples Leaf: {param_grid['min_samples_leaf']}")
        print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best Cross Validation F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_models['Decision Tree'] = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    

    def tune_random_forest(self):
        """Tune Random Forest Classifier with Grid Search"""
        print("\n")
        print("2. RANDOM FOREST CLASSIFIER - GRID SEARCH")
        print("="*80)
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTuning parameters:")
        print(f"  N Estimators: {param_grid['n_estimators']}")
        print(f"  Max Depth: {param_grid['max_depth']}")
        print(f"  Min Samples Split: {param_grid['min_samples_split']}")
        print(f"  Min Samples Leaf: {param_grid['min_samples_leaf']}")
        print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_models['Random Forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    

    def tune_knn(self):
        """Tune k-Nearest Neighbors with Grid Search"""
        print("\n")
        print("3. K-NEAREST NEIGHBORS - GRID SEARCH")
        print("="*80)
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'metric': ['euclidean', 'manhattan'],
            'weights': ['uniform', 'distance']
        }
        
        knn = KNeighborsClassifier()
        
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTuning parameters:")
        print(f"  N Neighbors: {param_grid['n_neighbors']}")
        print(f"  Metric: {param_grid['metric']}")
        print(f"  Weights: {param_grid['weights']}")
        print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search.fit(self.X_train_scaled, self.y_train)  # Using Scaled data
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_models['KNN'] = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    

    def tune_svm(self):
        """Tune Support Vector Machine with Grid Search"""
        print("\n")
        print("4. SUPPORT VECTOR MACHINE - GRID SEARCH")
        print("="*80)
        
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }
        
        svm = SVC(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nTuning parameters:")
        print(f"  Kernel: {param_grid['kernel']}")
        print(f"  C: {param_grid['C']}")
        print(f"  Gamma: {param_grid['gamma']}")
        print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_models['SVM'] = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    

    def evaluate_all_classifiers(self):
        """Evaluate all classifiers on training data and store results"""
        print("\n")
        print("EVALUATION ON TRAINING DATA\n")
        
        classifiers = ['Decision Tree', 'Random Forest', 'KNN', 'SVM']
        
        for clf_name in classifiers:
            clf = self.best_models[clf_name]
            
            if clf_name in ['KNN', 'SVM']:
                y_pred = clf.predict(self.X_val_scaled)
            else:
                y_pred = clf.predict(self.X_val)
            
            # Calculate metrics 
            precision = CustomMetrics.precision(self.y_val, y_pred, average='weighted')
            recall = CustomMetrics.recall(self.y_val, y_pred, average='weighted')
            specificity = CustomMetrics.specificity(self.y_val, y_pred, average='weighted')
            f1 = CustomMetrics.f1_score(self.y_val, y_pred, average='weighted')
            accuracy = CustomMetrics.accuracy(self.y_val, y_pred)
            
            self.results[clf_name] = {
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1,
                'Accuracy': accuracy
            }
            
            print(f"\n{clf_name}:")
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            print(f"  F1-Score:    {f1:.4f}")
            print(f"  Accuracy:    {accuracy:.4f}")
    
    def create_results_table(self):
        """Create and save results comparison table"""
        print("RESULTS COMPARISON TABLE")
        print(f"\n")
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['Precision', 'Recall', 'Specificity', 'F1-Score', 'Accuracy']]
        
        print("\n" + results_df.to_string())        
        return results_df
    
    def get_best_classifier(self):
        """Identify the best classifier based on F1-Score"""
        best_clf_name = max(self.results.keys(), 
                           key=lambda x: self.results[x]['F1-Score'])
        best_f1 = self.results[best_clf_name]['F1-Score']
        

        print("="*80)
        print(f"\n Best Classifier: {best_clf_name}")
        print(f" F1-Score: {best_f1:.4f}")
        
        return best_clf_name, self.best_models[best_clf_name]
    
    def predict_test_set(self, clf_name):
        """Generate predictions on test set"""
        print("\n")
        print("GENERATING TEST PREDICTIONS")
        print("\n")

        if self.X_test is None:
            print("\nError: No test file was provided during initialization.")
            return None
        
        clf = self.best_models[clf_name]
        
        # Use scaled data for KNN and SVM
        if 'Poly' in clf_name:
            X_test_final = self.X_test_poly
        elif clf_name in ['KNN', 'SVM']:
            X_test_final = self.X_test_scaled
        else:
            X_test_final = self.X_test

        y_pred = clf.predict(X_test_final)
        
        print(f"\n  Predictions generated for {len(y_pred)} test instances")
        print(f"  Unique classes in predictions: {np.unique(y_pred)}")
        print(f"  Class distribution: {dict(pd.Series(y_pred).value_counts().sort_index())}")
        
        return y_pred
    
    def save_predictions(self, predictions, filename):
        """Save predictions in txt file (one per line)"""
        with open(filename, 'w') as f:
            for pred in predictions:
                f.write(f"{int(pred)}\n")
        
        print(f"  Predictions saved: {filename}")


    def feature_transformation(self, degree=4, target_clf_name='Random Forest'):
        """Apply polynomial feature transformation to training data"""

        print(f"5. POLYNOMIAL FEATURE TRANSFORMATION (Degree {degree})")

        self.poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        self.X_train_poly = self.poly_transformer.fit_transform(self.X_train)
        self.X_val_poly = self.poly_transformer.transform(self.X_val)
        if self.X_test is not None:
            self.X_test_poly = self.poly_transformer.transform(self.X_test)  
        
        print(f"\n  Polynomial Features (Degree {degree}) applied.")
        print(f"  New feature count: {self.X_train_poly.shape[1]} features.")

        if target_clf_name in self.best_models and isinstance(self.best_models[target_clf_name], RandomForestClassifier):
            
            old_rf = self.best_models[target_clf_name]
            best_params = old_rf.get_params()
            
            # Create a new RF model with the best parameters found during the initial grid search
            new_rf = RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                criterion=best_params.get('criterion', 'gini'), # Ensure criterion is included
                random_state=self.random_state
            )
            
            # Train the new model on the polynomial features
            new_rf.fit(self.X_train_poly, self.y_train)
            
            # Store the new model under a new name
            poly_clf_name = f'{target_clf_name} (Poly D{degree})'
            self.best_models[poly_clf_name] = new_rf
            print(f"\n  Retrained '{poly_clf_name}' on transformed features.")

            # 3. Evaluate on Validation Set and Check for Perfection (using CustomMetrics)
            y_pred_poly = new_rf.predict(self.X_val_poly)
            
            f1_poly = CustomMetrics.f1_score(self.y_val, y_pred_poly, average='weighted')
            accuracy_poly = CustomMetrics.accuracy(self.y_val, y_pred_poly)
            

            print(f"\n  Validation Set F1-Score: {f1_poly:.4f}")
            print(f"  Validation Set Accuracy: {accuracy_poly:.4f}")

            if f1_poly == 1.0:
                print("\nSUCCESS: PERFECT CLASSIFICATION (F1=1.0) ACHIEVED on Validation Data!")
            else:
                print("\nNOTE: Perfect classification (F1=1.0) was NOT achieved on Validation Data. Consider increasing degree.")

           
            return poly_clf_name, new_rf
               
