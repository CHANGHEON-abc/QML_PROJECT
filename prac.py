import pickle as pkl
import pandas as pd

if __name__ == "__main__":
    with open("./training_dataset/oracle_edges.pkl", "rb") as f:
        dem_weights = pkl.load(f)
    with open("./training_dataset/oracle_logical_error_rates.pkl", "rb") as f:
        logical_error_rates = pkl.load(f)

    print(dem_weights[87][43])
    print(logical_error_rates[87])
    
    # dem_weights[n][m]
    # logical_error_rates[n]
    # n is index of number of datasets
    # m is edge index