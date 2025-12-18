# Datasheet: Black-Box Optimization Capstone Data Set

### Motivation
This data set was created to document the search process for the global optima of eight unknown black-box functions. The primary task is to support the development and evaluation of optimization strategies that can navigate high-dimensional spaces with extremely limited data samples.

### Composition
The data set contains input-output pairs for eight distinct functions (Function 1 through Function 8), ranging from 2D to 8D search spaces. Each entry includes the input vector, the specific kernel used in the surrogate model, the acquisition strategy (UCB or EI), the acquisition score, and the resulting function output. The data set currently consists of approximately 20 samples per function. A major gap exists in the high-dimensional functions (F6-F8), where the sample size is insufficient to fully characterize the complex landscape.

### Collection Process
Data was collected sequentially over ten iterations. Each week, a "Logic Engine" was used to evaluate 16 different hyperparameter configurations per function. The engine selected the query with the highest acquisition score based on the current internal surrogate model. These queries were then evaluated against the black-box functions to generate the next set of training data.

### Preprocessing and Uses
Inputs were preprocessed into hashable formats (tuples) to maintain consistency across the Python-based optimization loop. This data set is intended for benchmarking Bayesian Optimization algorithms and studying model behavior in sparse, high-dimensional environments. It is not suitable for modeling functions with high degrees of non-stationarity or sudden discontinuities without further modification.

### Distribution and Maintenance
The data set will be maintained by the project developer in a public GitHub repository. It will update weekly as new query results are integrated. The data is available for academic and research purposes under open-source licensing terms.
