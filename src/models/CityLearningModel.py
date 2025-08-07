from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pickle


def define_model():
    return RandomForestRegressor()

class CityLearningModel:
    def __init__(self):
        self.model = define_model()
        pass
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X, osmids): # TODO: Needs some work here
        '''
        osmids represent the list of osmids to be fed here
        '''
        y_pred = self.model.predict(X)
        y_pred = [round(x) for x in y_pred]

        X['osmid'] = osmids
        X['y_pred'] = y_pred

        def majority_vote(group):
            return group.mode().iloc[0]
    
        y_pred = X.groupby('osmid')['y_pred'].transform(majority_vote)
        return y_pred
    
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)        
        return mae, r2, mse, rmse











def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # %%

    model = define_model()
    model.fit(X_train, y_train)

    # %%
    y_pred = model.predict(X_test)

    # %%

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(mae, ', ', r2, ', ', mse, ', ', rmse)
    
    return mae, r2, mse, rmse