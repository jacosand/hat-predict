# About HATPredict

## The Problem of Selective C–H Functionalization

---

[**Selective C–H functionalization**](https://en.wikipedia.org/wiki/Carbon%E2%80%93hydrogen_bond_activation) is a cornerstone of modern organic synthesis, enabling the targeted conversion of **particular C–H bonds** into **carbon-heteroatom bonds like C–O or C–N** that are easy to manipulate subsequently.  The major challenge is finding a catalyst that reacts selectively with the desired C–H bond in a complex molecule from among the many other C–H bonds present.

Many selective C–H functionalizations begin with **hydrogen atom transfer (HAT) reactions**, in which the catalyst selectively grabs a particular hydrogen atom away from the substrate.

Consider the example below comparing how two different catalysts might react with the menthol molecule (responsible for the flavor of peppermint):

<center>
![Overview of selective C–H functionalization via hydrogen atom transfer (HAT)](/static/images/overview.svg)
</center>

In this example, the active radical forms of **two different catalysts** grab two different hydrogen atoms away from menthol to generate **two different radicals**.  These different radicals ultimately react further at the radical sites to form **two different final products**.  (In many reactions, different catalysts actually form different **distributions** of products rather each forming a single product.)

The purpose of HATPredict is to predict the outcomes in reactions like the ones shown above.  Consider a general **hydrogen atom transfer (HAT) reaction**

<center>
### A·&nbsp;&nbsp;&nbsp;+&nbsp;&nbsp;&nbsp;B–H&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp;&nbsp;A–H&nbsp;&nbsp;&nbsp;+&nbsp;&nbsp;&nbsp;B·
</center>

where **A•** is the active radical form of the catalyst and **B–H** represents the substrate molecule.

The **problem solved by HATPredict** is to predict the distribution of radical products **B•** that result from the reaction, with different products **B•** corresponding to the catalyst reacting with different C–H bonds in the substrate molecule **B–H**.

## Traditional Density Functional Theory Approach

---

The usual way to predict the distribution of products is with **density functional theory (DFT)** computations of all the **transition states [A--H--B]<sup>‡</sup>** that lead to different products.  For even a moderately complex reaction, this approach takes **weeks to months**, requiring the hands-on expertise of a computational chemist and dedicated access to a computing cluster.  This makes it unsuitable for high-throughput screening of catalysts.

## HATPredict Machine Learning Approach

---

**HATPredict** uses a trained [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) machine learning model to predict the distribution of radical products in hydrogen atom transfer reactions.

The HATPredict model uses input features computed in **seconds to minutes** on an [Amazon Web Services (AWS) EC2](https://aws.amazon.com/ec2/) web server.  These input features are computed **in real time** whenever a user makes an inference call to the model.

HATPredict is trained on **free energy barriers (∆G<sup>‡</sup>)** for **2926 HAT reactions** computed with density functional theory in this [paper](https://doi.org/10.1039/D1QO01325D).

While the model described in the paper is a major advance because it does not require any transition states at inference time, it still requires density functional theory computations on the starting materials and products (**A–H**,  **A•**, **B–H**, and **B•**) to generate input features.

In contrast, HATPredict does not require any density functional theory computations at inference time.  This step forward is what enables HATPredict to deliver real-time results in **seconds to minutes** to users without any computational chemistry knowledge.

HATPredict model training is described and performed in this [Jupyter notebook](https://github.com/jacosand/hat-predict/blob/main/model/training/model_training.ipynb), and involves a Bayesian optimization hyperparameter search using [Hyperopt](http://hyperopt.github.io/hyperopt/).

## HATPredict Model Real-Time Input Features

---

All input features are obtained exclusively from real-time computations performed on the ground-state reacting compounds (**A–H**,  **A•**, **B–H**, and **B•**) as follows:

1.  Molecular geometries are obtained via a conformational search and force field optimization in [OpenBabel](https://openbabel.org/wiki/Main_Page).

2.  Geometries are further optimized via [Semiempirical Tight Binding (xTB)](https://xtb-docs.readthedocs.io/en/latest/contents.html) computations.  These computations also generate electronic, charge, and bond order input features.

3.  Bond dissociation energies and bond dissociation free energies are also required as input features.  These are obtained by making inference calls to the [alfabet](https://bde.ml.nrel.gov/) graph convolutional neural network model described in this [paper](https://doi.org/10.1038/s41467-020-16201-z).

4.  Steric input features are obtained via the [morfeus](https://kjelljorner.github.io/morfeus/) software package.

## HATPredict Model Performance

---

The graph below illustrates the performance of HATPredict on the test set from this [paper](https://doi.org/10.1039/D1QO01325D).

<center>
![Performance of HATPredict compared to traditional transition state computations with density functional theory)](/static/images/performance.png)
</center>

The graph compares the free energy barriers (∆G<sup>‡</sup> in kcal/mol) calculated by HATPredict (predicted labels) to those obtained in the traditional way via the computation of transition states with density functional theory (ground-truth labels).

As the graph shows, most free energy barriers are well-predicted by HATPredict to within 1.4 kcal/mol (the points fall within the yellow band on the graph), which is a commonly-used benchmark for determining whether a new method is chemically accurate.  The mean average error is smaller, 0.98 kcal/mol.

It is important to note that this test set uses radicals (**A•**) present in the training data reacting with new substrates (**B–H**) that are not present in the training data, so it only reflects the ability of the model to generalize to new substrates.  However, most radicals (**A•**) used in selective C–H functionalization reactions fall into a few groups with input features similar to one or more of the radicals already present in the training data.  Nevertheless, I am currently working to expand the training data to more radicals and thoroughly evaluate the ability of the model to generalize to unseen radicals.

## Further Development

---

I am currently working to develop a **graph convolutional network (GCN)** model with the goal of eliminating the explicit computation of input features and decreasing the inference time.

I am also working to **expand the training data** to a larger number of input radicals (**A•**) with the goal of improving model performance on new and unseen radicals.