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
5. Print Cost Plot
6. Predict

The level of abstraction was deliberately chosen for better code clarity. I explicitly showed 'for' loops within the main code (outside of functions) to explicitly show the flow of operations throughout the network. [Helper Functions](helper_functions.py) were written to provide the abstraction. Also, I wrote a *scratch* [gradient checker](gradient_checking.py) to debug initial errors in my backprop implementation.

Lastly, with 1500 iterations, I've got the following results: 

