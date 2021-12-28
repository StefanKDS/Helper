import matplotlib as pyplot


def linear_regression_feature_importance(model):
    importance = model.coef_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def logistic_regression_feature_importance(model):
    importance = model.coef_[0]
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def cart_regression_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def cart_classification_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def random_forest_regression_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def random_forest_classification_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def xgboost_regression_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def xgboost_classification_feature_importance(model):
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()