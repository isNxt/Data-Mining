# 实验五 集成学习与特征选择

## 实验目的
- 使用集成学习系统练习，体会集成学习系统对分类性能的影响。
- 学习scikit-learn 模块的用法，练习使用模块中的基本分类器解决问题。
- 了解对有监督学习，特征选择对分类结果的影响。

## 实验数据
- 数据集为Microarray数据，具有高维度小样本的特点。
- 所有数据需转置，转换成样本*特征的方式。
- 转置之后，第一列为类别标签。

## 实验内容
- 使用多种不同分类器，比较在基分类器参数设定一致的情况下，使用不同基分类器，不同集成学习方法，获得的结果的不同，以及与单个分类器获得结果的区别，并进行分析；
- 请用soft、hard方法分别集成多个不同的基本分类器，比较结果的区别；
- 通过结合特征选择方法，分析特征选择对实验结果的影响。

## 实验笔记
### Algorithm
- Bagging: create classifiers using training sets that are bootstrapped (drawn with replacement)
    - Decision Forest：randomly select part of features;
    - Random Forest：randomly select both samples and features;
    - Rotation Forest：make a new sample by projecting the original to a new subspace;
- Boosting
    - Adboost
- Other Frameworks
    - Mixture of Experts
    - Stacking
    - Cascading
### Evaluate
- Kappa-error Diagram
- Diversity Measurements

### Package
- [sklearn.ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
- [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
    sklearn.ensemble: the sklearn.ensemble module includes ensemble-based methods for classification, regression and anomaly detection.

    User guide: See the Ensemble methods section for further details.
        ensemble.AdaBoostClassifier([…])	An AdaBoost classifier.
        ensemble.BaggingClassifier([base_estimator, …])	A Bagging classifier.
        ensemble.ExtraTreesClassifier([…])	An extra-trees classifier.
        ensemble.GradientBoostingClassifier([loss, …])	Gradient Boosting for classification.
        ensemble.IsolationForest([n_estimators, …])	Isolation Forest Algorithm
        ensemble.RandomForestClassifier([…])	A random forest classifier.
        ensemble.RandomTreesEmbedding([…])	An ensemble of totally random trees.
        ensemble.VotingClassifier(estimators[, …])	Soft Voting/Majority Rule classifier for unfitted estimators.

    Partial dependence: Partial dependence plots for tree ensembles.
        ensemble.partial_dependence.partial_dependence(…)	Partial dependence of target_variables.
        ensemble.partial_dependence.plot_partial_dependence(…)	Partial dependence plots for features.
        sklearn.exceptions: Exceptions and warnings

Hard Voting Classifier：根据少数服从多数来定最终结果；
Soft Voting Classifier：将所有模型预测样本为某一类别的概率的平均值作为标准，概率最高的对应的类型为最终的预测结果；