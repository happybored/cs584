1. multiclass_classfication.py:
    just change whatever you want

2. feature_engineering.py:
    ## just change the following step:
    X = np.concatenate([X], axis=1)  #original
    #X = np.concatenate([X_parity], axis=1)  #parity
    #X = np.concatenate([X_lde], axis=1)   #lde
    X = np.concatenate([Xb], axis=1)  #binary encoding
    #X = np.concatenate([X_parity，X_lde], axis=1)  #combinations
    #X = np.concatenate([X,X_parity，X_lde], axis=1) 

3. path_generation.py
    ## Test the model by generating a path
    env = SO6Env(t_step= 10,initial='random')  # LUT[:10]
    #env = SO6Env(t_step= 10,initial= None)   # Random 10

