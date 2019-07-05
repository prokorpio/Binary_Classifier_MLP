# Binary Classifier MLP

As an exercise I built a simple neural network from scratch using Numpy. This'll be classifying whether or not there's a cat in a photo. 

The [main code](binary_classifier_MLP.py) has multiple sections: 
1. Load Dataset
2. Dataset Preprocessing
    - Flatten
    - Normalize
3. Train Model
    - Forward Prop
    - Compute Cost
    - Backward Prop
    - Gradient Descent
4. Print Train Set Accuracy
5. Predict



The level of abstraction was deliberately chosen for better code clarity. I explicitly showed `for` loops within the main code (outside of functions) to explicitly show the flow of operations throughout the network. [Helper functions](helper_functions.py) were written to provide the abstraction. Also, I wrote a *scratch* [gradient checker](gradient_checking.py) to debug initial errors in my backprop implementation.

Lastly, with 1500 iterations and with vanilla setup (no regularization etc.), I've got the following results: 

```
$ python3 binary_classifier_MLP.py

    Cost after iteration 0: 0.7717493284237686
    Cost after iteration 150: 0.6623642404728758
    Cost after iteration 300: 0.6115068816101356
    Cost after iteration 450: 0.5610104976233579
    Cost after iteration 600: 0.5279299569455267
    Cost after iteration 750: 0.4375577687942258
    Cost after iteration 900: 0.39174697434805356
    Cost after iteration 1050: 0.3122831979508355
    Cost after iteration 1200: 0.23741853400268137
    Cost after iteration 1350: 0.19720003039541167
    Train Set Accuracy =  0.9808612440191385
    Test Set Accuracy = 0.8200000000000001'
```
