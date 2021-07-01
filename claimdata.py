import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold

def load_claim_data(pathX, pathY, k, load=False):
    print('Importing dataset ...')

    if load:
        datasets = pickle.load(open(load, "rb"))
        return datasets

    # Import data as numpy arrays
    x = pd.read_csv(pathX).to_numpy()
    x = np.concatenate((x, np.expand_dims(np.arange(x.shape[0]),1)), axis=1)

    y = pd.read_csv(pathY).to_numpy()

    # Get array of unique providers
    providers_data = pd.DataFrame(y[:,1:]).drop_duplicates().to_numpy()

    # Create bags
    print('Creating bags ...')

    bags_fea = []

    provider_codes = list(y[:,1])
    instance_indeces = list(y[:,0])

    for i, provider in enumerate(providers_data[:,0]):
        print('Creating Bag of Provider {0:6d}'.format(i))
        instance_indeces = [i for i, p in enumerate(provider_codes) if p == provider]
        matrix = x[instance_indeces, 1:]
        vector = y[instance_indeces, 2]
        bags_fea.append((matrix, list(vector)))
    
    bag_idxs = np.arange(len(bags_fea))         
    bag_cut = int(np.floor(len(bags_fea) * 0.80))

    train_idxs = bag_idxs[:bag_cut]
    test_idxs = bag_idxs[bag_cut:]

    bags_fea_train = [bags_fea[ibag] for ibag in train_idxs]
    bags_fea_test = [bags_fea[ibag] for ibag in test_idxs]

    # KFold - split train into train and validation
    print('Creating folds ...')

    datasets = []

    if k == 1:
        idxs = np.arange(len(bags_fea_train)) 
        np.random.shuffle(idxs)
        
        cut = int(np.floor(len(bags_fea_train) * 0.80))

        train_idxs = idxs[:cut]
        test_idxs = idxs[cut:]
        
        dataset = {}
        dataset['train'] = [bags_fea_train[ibag] for ibag in train_idxs]
        # validation. we call it test because then it is referenced as test
        dataset['test'] = [bags_fea_train[ibag] for ibag in test_idxs]
        datasets.append(dataset)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=None)
        for train_idx, test_idx in kf.split(bags_fea_train):
            dataset = {}
            dataset['train'] = [bags_fea_train[ibag] for ibag in train_idx]
            # validation. we call it test because then it is referenced as test
            dataset['test'] = [bags_fea_train[ibag] for ibag in test_idx]
            datasets.append(dataset)

    # Convert test into needed format
    dataset_test = {
        'test': bags_fea_test
    }

    print('Data imported succesfully')

    # Save
    pickle.dump(datasets, open("datasets_train_val.pkl", "wb"))
    pickle.dump(dataset_test, open("dataset_test.pkl", "wb"))

    print('Data pickled')

    return datasets