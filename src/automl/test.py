import re

a = "MyDummyClassifier(configuration=1, init_params=None, random_state=None)"

b = "SimpleClassificationPipeline({'balancing:strategy': 'weighting', 'classifier:__choice__': 'gradient_boosting', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'no_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'minority_coalescer', 'data_preprocessing:numerical_transformer:imputation:strategy': 'mean', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'standardize', 'feature_preprocessor:__choice__': 'select_rates', 'classifier:gradient_boosting:early_stop': 'valid', 'classifier:gradient_boosting:l2_regularization': 0.787172957129578, 'classifier:gradient_boosting:learning_rate': 0.23076913534674612, 'classifier:gradient_boosting:loss': 'auto', 'classifier:gradient_boosting:max_bins': 255, 'classifier:gradient_boosting:max_depth': 'None', 'classifier:gradient_boosting:max_leaf_nodes': 8, 'classifier:gradient_boosting:min_samples_leaf': 4, 'classifier:gradient_boosting:scoring': 'loss', 'classifier:gradient_boosting:tol': 1e-07, 'data_preprocessing:categorical_transformer:category_coalescence:minority_coalescer:minimum_fraction': 0.002842817334543296, 'feature_preprocessor:select_rates:alpha': 0.2779207466036798, 'feature_preprocessor:select_rates:mode': 'fwe', 'feature_preprocessor:select_rates:score_func': 'f_classif', 'classifier:gradient_boosting:n_iter_no_change': 10, 'classifier:gradient_boosting:validation_fraction': 0.1}, dataset_properties={'task': 1, 'sparse': False, 'multilabel': False, 'multiclass': False, 'target_type': 'classification', 'signed': False})"


def get_pipe(pipe_str):
    outer = re.compile("\(.+\)")
    m = outer.search(pipe_str)
    innerre = re.compile("\{(.+)},{(.+)\}")
    regex = r"\{(.*?)\}"
    matches = re.finditer(regex, m.group(), re.MULTILINE | re.DOTALL)
    for match in matches:
        print("\n")
        x =eval("{" + match.group(1) + "}")
        try:
            fmodel = x['classifier:__choice__']
        except:
            fmodel = None
    return fmodel

def get_fmodel(fmodel):
    fensemble = dict()
    for n in fmodel:
        fensemble[get_pipe(n[1])] = n[0]


    return fmodel
if __name__ == "__main__":
    get_pipe
    # inter = re.compile("\{.+\}")
    # n = innerre.search(m.group()[1:-1])
    # print(n)