import pickle as pkl

if __name__ == "__main__":
    with open("dem_weights_d_3.pkl", "rb") as f:
        dem_weights = pkl.load(f)
    
    # Weights for static & oracle methods
    print(dem_weights['configurations'][0]['decoders']['static']['edges'][81])
    print(dem_weights['configurations'][0]['decoders']['oracle']['edges'][576])
    
