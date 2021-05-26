import numpy as np

def load_feature(datasetPath = 'D:/DIP_data/test_file/'):
    load_hsv_Xtest = np.loadtxt(datasetPath + "hsv_feather_Xtest.txt")
    load_hsv_Xtrain = np.loadtxt(datasetPath + "hsv_feather_Xtrain.txt")

    load_histogram_Xtrain = np.loadtxt(datasetPath + "histogram_feather_Xtrain.txt")
    load_histogram_Xtest = np.loadtxt(datasetPath + "histogram_feather_Xtest.txt")

    feature_train = np.append(load_hsv_Xtrain, load_histogram_Xtrain, axis=1)
    feature_test = np.append(load_hsv_Xtest, load_histogram_Xtest, axis=1)
    # print(feature_train)
    return feature_train, feature_test

def load_labels():
    from data_utils import load_CIFAR10
    #加载Cifar10数据集，并输出数据集的维数
    cifar10_dir = 'D:\DIP_project\cifar-10-python\cifar-10-batches-py'
    X_train,y_train,X_test,y_test = load_CIFAR10(cifar10_dir) # Y是标签（0-9）
    return y_train, y_test

if __name__ == '__main__':
    train_feature, test_feature = load_feature()
    # print(train_feature)
    train_label,test_label = load_labels()
    '''
    # random forest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(train_feather, train_label)
    score_c = clf.score(test_feature, test_label)
    print(score_c)
    test_pre = clf.predict(test_feature)
    pre_log = clf.predict_log_proba(test_feature[0:1,:])
    print(pre_log)
'''
    # MLP
    from sklearn.neural_network import MLPClassifier 
    clf_MLP = MLPClassifier(random_state=1, max_iter=300).fit(train_feature, train_label)
    score_MLP = clf_MLP.score(test_feature, test_label)
    print(score_MLP)

    # print(clf.predict(test_feature[0:1,:]))

    #save model
    from sklearn.externals import joblib
    joblib.dump(clf_MLP,'clf_MLP.model')
'''
    #load model
    clf_MLP_load = joblib.load('clf_MLP.model')
    # print(clf_MLP.predict(test_feature[0:1,:]))
    score_load = clf_MLP_load.score(test_feature, test_label)
    print(score_load)
'''






'''
    print(clf.predict(test_feature[0:1,:]))
    #save model
    from sklearn.externals import joblib
    joblib.dump(clf,'clf.model')
    #load model
    clf1 = joblib.load('clf.model')
    print(clf1.predict(test_feature[0:1,:]))
'''
