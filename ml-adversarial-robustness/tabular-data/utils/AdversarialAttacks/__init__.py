from tabnanny import verbose
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, BoundaryAttack, HopSkipJump 
import numpy as np

# Paper link: https://arxiv.org/abs/1708.03999
def ZooAttackEvaluation(model, X_test, y_test, k = 5):

    art_model = SklearnClassifier(model = model)

    results = []
    for _ in range(k):

        # Create ART Zeroth Order Optimization attack
        zoo = ZooAttack(
            classifier = art_model,
            binary_search_steps = 10,
            nb_parallel = 1,
            learning_rate = 0.01,
            max_iter = 20,
            abort_early = True,
            variable_h=0.2,
            verbose = False
        )

        # Generate adversarial samples with ART Zeroth Order Optimization attack
        X_adv = zoo.generate(X_test)

        # Evaluation
        y_pred_adv = model.predict(X_adv)
        acc = accuracy_score(y_test, y_pred_adv)

        # Robust Accuracy
        results.append(acc)

    return np.array(results)

# Paper link: https://arxiv.org/abs/1712.04248
def BoundaryAttackEvaluation(model, X_test, y_test, k = 5):

    art_model = SklearnClassifier(model = model)

    results = []
    for _ in range(k):

        attack = BoundaryAttack(
            estimator = art_model,
            targeted = False,
            max_iter = 20,
            delta = 0.01,
            epsilon = 0.01,
            verbose = False
        )

        X_adv = attack.generate(X_test)
        
        # Evaluation
        y_pred_adv = model.predict(X_adv)
        acc = accuracy_score(y_test, y_pred_adv)

        # obust Accuracy
        results.append(acc)

    return np.array(results)

# Paper link: https://arxiv.org/abs/1904.02144
def HopSkipJumpEvaluation(model, X_test, y_test, k = 5):

    art_model = SklearnClassifier(model = model)

    results = []
    for _ in range(k):

        attack = HopSkipJump(
            classifier  = art_model,
            targeted = False,
            max_iter = 20,
            verbose = False
        )
        
        X_adv = attack.generate(X_test)
        
        # Evaluation
        y_pred_adv = model.predict(X_adv)
        acc = accuracy_score(y_test, y_pred_adv)

        # return Robust Accuracy
        results.append(acc)
    
    return np.array(results)