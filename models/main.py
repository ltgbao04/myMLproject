import visualize
from utils import *
import train

if __name__ == "__main__":
    df = load_data("../data/segmented_customers.csv")
    X_train, Y_train, models = train.trainModel(df)
    visualize.visualize(X_train, Y_train, models)