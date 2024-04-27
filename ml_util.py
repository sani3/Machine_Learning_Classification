from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SplitData(BaseEstimator, TransformerMixin):
    """A custorm transformer to split data set into training and test sets"""
    def __init__(self, strat=True):
        self.strat = strat
        self.y = None
        self.spliter = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

    def fit(self, X, y):
        self.y = y
        return self

    def transform(self, X, y=None):
        if self.strat == True:
            self.spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  
            for train_index, test_index in self.spliter.split(X, self.y):
                self.train_x = X[train_index, :]
                self.test_x = X[test_index, :]
                self.train_y = self.y[train_index]
                self.test_y = self.y[test_index]
        else:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, self.y, test_size=0.2, random_state=42)
        return self.train_x, self.test_x, self.train_y, self.test_y



class ScaleFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, standard=False):
        self.standard = standard
        self.scaler = None
    
    def fit(self, X, y=None):
        if self.standard == True:
            self.scaler = StandardScaler()
            return self
        else:
            self.scaler = MinMaxScaler()
            return self
    
    def transform(self, X, y=None):
        data = self.scaler.fit_transform(X) 
        return data
        

def train_accuracy(train_y, train_y_pred):
    """Returns training accuracy"""
    accuracy = accuracy_score(train_y, train_y_pred)
    return {"Training Accuracy": accuracy}

def cv_accuracy(model, train_x, train_y):
    """Returns cross-validation accuracies"""
    cv_accuracies = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=5)
    return {
        "CV accuracies": cv_accuracies, 
        "Mean": cv_accuracies.mean(), 
        "Standard deviation": cv_accuracies.std()
    }
