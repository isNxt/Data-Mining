from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib as plt

import utils



def test():
    # define datasets-----------------------------------
    datasets = ['Breast']
    names = ["DecisionTree", "KNeighbors","GaussianNB"]
    # define classifiers-------------------------------------------
    classifiers = [DecisionTreeClassifier(max_depth=4), KNeighborsClassifier(n_neighbors=3), GaussianNB()]
    clfs = list(zip(names, classifiers))
    eclf_soft = VotingClassifier(estimators=clfs, voting='soft')
    eclf_hard = VotingClassifier(estimators=clfs, voting='soft')
    classifiers.append(eclf_soft)
    classifiers.append(eclf_hard)
    names.append("VotingSoft")
    names.append("VotingHard")

    # iterate over datasets
    for dataset in datasets:
        X_train, y_train = utils.read_data('./data/'+dataset+'_train.data')
        X_test, y_test = utils.read_data('./data/'+dataset+'_test.data')
        # iterate over classifiers-------------------------------------------
        probas = []
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            # predict class probabilities for all classifiers
            probas.append(clf.predict_proba(X_test))
        # get class probabilities for the first sample in the dataset
        class1_1 = [pr[0, 0] for pr in probas]
        class2_1 = [pr[0, 1] for pr in probas]
        class3_1 = [pr[0, 2] for pr in probas]
        class4_1 = [pr[0, 3] for pr in probas]
        class5_1 = [pr[0, 4] for pr in probas]
        # plotting

        N = 4  # number of groups
        ind = np.arange(N)  # group positions
        width = 0.35  # bar width

        fig, ax = plt.subplots()
        # bars for classifier 1-3
        p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
                    color='green', edgecolor='k')
        p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
                    color='lightgreen', edgecolor='k')

        # bars for VotingClassifier
        p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
                    color='blue', edgecolor='k')
        p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
                    color='steelblue', edgecolor='k')

        # plot annotations
        plt.axvline(2.8, color='k', linestyle='dashed')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(['LogisticRegression\nweight 1',
                            'GaussianNB\nweight 1',
                            'RandomForestClassifier\nweight 5',
                            'VotingClassifier\n(average probabilities)'],
                           rotation=40,
                           ha='right')
        plt.ylim([0, 1])
        plt.title('Class probabilities for sample 1 by different classifiers')
        plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
        plt.tight_layout()
        plt.show()

def main():
    # define datasets-----------------------------------
    datasets = ['Breast','GCM','Leukemia1','Leukemia2']
    names = ["DecisionTree", "KNeighbors","GaussianNB"]
    # define classifiers-------------------------------------------
    classifiers = [DecisionTreeClassifier(max_depth=4), KNeighborsClassifier(n_neighbors=3), GaussianNB()]
    clfs = list(zip(names, classifiers))
    eclf_soft = VotingClassifier(estimators=clfs, voting='soft')
    eclf_hard = VotingClassifier(estimators=clfs, voting='soft')
    classifiers.append(eclf_soft)
    classifiers.append(eclf_hard)
    names.append("VotingSoft")
    names.append("VotingHard")

    # iterate over datasets
    for dataset in datasets:
        X_train, y_train = utils.read_data('./data/'+dataset+'_train.data')
        X_test, y_test = utils.read_data('./data/'+dataset+'_test.data')
        # iterate over classifiers-------------------------------------------
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)


if __name__ == '__main__':
    test()
    main()
