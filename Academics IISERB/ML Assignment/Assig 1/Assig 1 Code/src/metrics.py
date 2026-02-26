import numpy as np
import pandas as pd

# This file contains the code for custom metrics for model evaluation

class CustomMetrics:
    """Custom implementation of classification metrics."""
    
    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred, classes=None):
        """Calculate confusion matrix manually """
        
        if classes is None:
            classes = np.unique(y_true)
        
        cm = {}
        
        for cls in classes:
            # Binary classification for this class (One-vs-Rest)
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            # True Positives: predicted correctly for a class
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            
            # False Positives: predicted wrongly as belongs to a class, but not belongs to that class
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            
            # True Negatives: predicted correctly as not belongs to a class
            tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
            
            # False Negatives: predicted wrongly as not belongs to a class, but actually belongs to that class
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            
            cm[cls] = {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            }
        
        return cm
    
    
    @staticmethod
    def precision(y_true, y_pred, classes=None, average='weighted'):
        """
        Precision = TP / (TP + FP)
        It tells : What proportion of predicted positives are actually positive?
        """
        if classes is None:
            classes = np.unique(y_true)
        
        cm = CustomMetrics.calculate_confusion_matrix(y_true, y_pred, classes)
        
        all_precisions = {}
        for cls in classes:
            tp = cm[cls]['TP']
            fp = cm[cls]['FP']
            
            if (tp + fp) == 0:
                all_precisions[cls] = 0
            else:
                all_precisions[cls] = tp / (tp + fp)
        
        if average == 'weighted':
            class_counts = np.bincount(y_true, minlength=max(classes) + 1)
            weighted_precision = sum(all_precisions[cls] * class_counts[cls] 
                                    for cls in classes) / len(y_true)
            return weighted_precision
        
        return all_precisions
    


    @staticmethod
    def recall(y_true, y_pred, classes=None, average='weighted'):
        """
        Precision = TP / (TP + FN)
        It tells : What proportion of actual positives are predicted positive?
        """
        if classes is None:
            classes = np.unique(y_true)
        
        cm = CustomMetrics.calculate_confusion_matrix(y_true, y_pred, classes)
        
        all_recalls = {}
        for cls in classes:
            tp = cm[cls]['TP']
            fn = cm[cls]['FN']
            
            if (tp + fn) == 0:
                all_recalls[cls] = 0
            else:
                all_recalls[cls] = tp / (tp + fn)
        
        if average == 'weighted':
            class_counts = np.bincount(y_true, minlength=max(classes) + 1)
            weighted_recall = sum(all_recalls[cls] * class_counts[cls] 
                                    for cls in classes) / len(y_true)
            return weighted_recall
        
        return all_recalls
    


    @staticmethod
    def sensitivity(y_true, y_pred, classes=None, average='weighted'):
        """
        Sensitivity = TP / (TP + FN)
        It is the True Positive Rate. (same as Recall)
        """
        return CustomMetrics.recall(y_true, y_pred, classes, average)
    

    
    @staticmethod
    def specificity(y_true, y_pred, classes=None, average='weighted'):
        """
        Specificity = TN / (TN + FP)
        It is the True Negative Rate
        """
        if classes is None:
            classes = np.unique(y_true)

        cm = CustomMetrics.calculate_confusion_matrix(y_true, y_pred, classes)

        all_specificities = {}
        for cls in classes:
            tn = cm[cls]['TN']
            fp = cm[cls]['FP']

            if tn + fp == 0:
                all_specificities[cls] = 0
            else:
                all_specificities[cls] = tn / (tn + fp)

        if average == 'weighted':
            class_counts = np.bincount(y_true, minlength=max(classes)+1)
            weighted_specificity = sum(all_specificities[cls] * class_counts[cls] for cls in classes) / (len(y_true))
            return weighted_specificity
        
        return all_specificities



    @staticmethod
    def f1_score(y_true, y_pred, classes=None, average='weighted'):
        """
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        It is the Harmonic mean of precision and recall
        """
        if classes is None:
            classes = np.unique(y_true)
        
        precisions = CustomMetrics.precision(y_true, y_pred, classes, average='macro')
        recalls = CustomMetrics.recall(y_true, y_pred, classes, average='macro')
        
        f1_scores = {}
        for cls in classes:
            p = precisions[cls]
            r = recalls[cls]
            
            if (p + r) == 0:
                f1_scores[cls] = 0
            else:
                f1_scores[cls] = 2 * (p * r) / (p + r)
        
        if average == 'weighted':
            class_counts = np.bincount(y_true, minlength=max(classes) + 1)
            weighted_f1 = sum(f1_scores[cls] * class_counts[cls] for cls in classes) / len(y_true)
            return weighted_f1
        
        return f1_scores
    
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Accuracy = (TP + TN) / Total
        Proportion of correct predictions
        """
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total