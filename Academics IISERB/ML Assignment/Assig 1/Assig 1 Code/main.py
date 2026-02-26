# RUN THIS MAIN FILE TO EXECUTE THE ENTIRE PIPELINE

import sys
import os
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from src.classifiers import *
from src.data_visualization import *

# We have defined the variable to handle the train and test file
train_data_csv_file = 'data/Training dataset.csv'
test_data_csv_file = 'data/Test data.csv'


# loading the data and visualizing it 
# (For the code refer to src/data_visualization.py)
df_train, feature_cols = load_and_explore_data(train_data_csv_file)
visualize_data(train_data_csv_file)



# Custom metrics are defined in the file - 'src/metrics.py' )



# Initializing the pipeline to find the best classifier for our data
pipeline = All_Classifiers(
    train_file=train_data_csv_file,
    test_file=test_data_csv_file
)

# Tuning all the  classifiers
print("STARTING GRID SEARCH FOR ALL CLASSIFIERS")

dt_model, dt_params, dt_score = pipeline.tune_decision_tree()
rf_model, rf_params, rf_score = pipeline.tune_random_forest()
knn_model, knn_params, knn_score = pipeline.tune_knn()
svm_model, svm_params, svm_score = pipeline.tune_svm()

# Evaluate all the tuned classifiers
pipeline.evaluate_all_classifiers()

# Creating the results table
results_df = pipeline.create_results_table()

# finding the best classifier
best_clf_name, best_clf = pipeline.get_best_classifier()



# Generating test predictions for test dataset
if pipeline.X_test is not None:
    print("\n GENERATING INITIAL TEST PREDICTIONS (Best Base Classifier)")
    test_predictions = pipeline.predict_test_set(best_clf_name)
    pipeline.save_predictions(test_predictions, 'predictions_best_classifier.txt')

    print("\n TEST SET PREDICTED SUCCESSFULLY! and saved at - 'predictions_best_classifier.txt'")
    print("="*80)




# Let's implement the Features Transformation to perfectly fit the train data to the best classifier i.e. Random Forest
print("\n STARTING FEATURE TRANSFORMATION USING POLYNOMIAL FEATURES")
best_poly_name, _ = pipeline.feature_transformation(degree=3, target_clf_name=best_clf_name)

# Final Prediction using the Polynomial Model
if best_poly_name:
    final_predictions = pipeline.predict_test_set(best_poly_name)
    pipeline.save_predictions(final_predictions, 'predictions_feature_transform.txt')
    print(" FINAL TEST SET PREDICTED SUCCESSFULLY! and saved at - 'predictions_feature_transform.txt'")
    print("="*80)


