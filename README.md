# CNN_Features_Extractor
>   Bioinformatics Project Polito

## Getting Started

### Directory Structure of the Repository

-   **Docu:** Contains the following files:
    -   *Report.pdf:* Latex report of the whole project.
    -   *Presentation.pptx:* PowerPoint Slides of project project.
-   **LaTeX_Report:** Latex files of the *Report.pdf*
-   **images:** The images dataset of the project useful for most programs.
-   *pathFolder.py:* Build the *.tsv* file of the whole images dataset.
-   *extract5.py:* Extract any (intermediate) layer of a pre-trained model for *Keras*.
-   *techniques.py* Useful for two purposes:
    -   *1:* Program for the algorithm TSNE or statistical procedure PCA visualization.
    -   *2:* Program to get the optimal number of features, concerning a given layer.
-   *KNN.py:* Implementation of a supervised learning *k-nearest neighbors algorithm* wich is a non-parametric method used for classification and regression.
-   *SVM.py:* Implementation of a supervised learning *support vector machines* model that analyze data used for classification and regression analysis.
-   *Bag_of_Features.m:* Example built in matlab, in which shows how to use a bag of features approach for image category classification.

### Command-Line Syntax for the _.py_ Progrmas

The requirements are the following:
>   Activate tensorflow before the run .py programs, ex. `source tensorflow/bin/active`

>   The images must be in the _PATH_ `./images/`

#### _pathFolder.py_

```
(tensorflow)$ python3 pathFolder.py
```   
>*output:* 
>-   Create a `file.tsv` with a _id_, _rootPATH_ and _label_ of each image, from the whole images dataset.

#### _extract5.py_

```
(tensorflow)$ python3 extract5.py <source file> <pre-trained model>
```

>*options:*
>-   *source file:*  _.tsv_ Images dataset, most cases `file.tsv`
>-   *pre-trained model:* These are the following pre-trained model for *Keras*, that you can choose
>     -   *VGG16*
>     -   *VGG19*
>     -   *DenseNet*
>     -   *MobileNet*
>     -   *ResNet50*
>     -   *Xception*
>     -   *NASNet*
>-   *Number of (intermediate)Layer:* While running, will be requested the number of the layer that you have to choose from a terminal list previously given.
    
>*output:*
>-   Create a _.tsv_ file with all the features up to the chosen layer; the format is the following `<pre-trained model>_<number layer chosen>.tsv`.

#### _techniques.py_

```
(tensorflow)$ python3 techniques.py <source file> <technique>
```

>Note: There is no _entry_ in which you can choose the number of components based on PCA descomposition, because the program calculates the optimal number of components for a certain layer (the percentage of explained variance must be at least 99.5%).

>*options:*
>-   *source file:*  _.tsv_ Features of the Images dataset in a certain layer, belonging to a pre-trained model ex. `VGG16_5.tsv`
>-   *technique:* These are the following dimensionality reduction methods for visualization.
>     -   *PCA*
>     -   *TSNE*
>     -   *PCA-TSNE*

>*output:* 
>-   Display in a terminal screen the optimal number of components for the PCA descomposition.
>-   Graph in a new figure window the requested method.

#### _KNN.py_

```
(tensorflow)$ python3 KNN.py <source file> [-n | --components n_comp] [-k | --neighbors n_nbr] [-s | --split s_number]
```

>*options:*
>-   *source file:*  _.tsv_ Features of the Images dataset in a certain layer, belonging to a pre-trained model ex. `VGG16_5.tsv`
>-   *n_comp:*  Number of components for thin K-Nearest Neighbor algorithme PCA descomposition, ex.`-n 75`, default `50`
>-   *n_nbr:*  Parameter _K_ (number of neighbors) in _K-Nearest Neighbor_ algorithm, ex.`-s 11`, default `5`
>-   *s_number:*  Number with which the database is divided into training and test sets (size of training set), ex.`-s 0.5`, default `0.7`

>*output:* 
>-   Display in a terminal screen the Confusion Matrix for the supervised learning algorithm K-NN.
>-   Display in a terminal screen the Average Accuracy for the supervised learning algorithm K-NN.

#### _SVM.py_


```
(tensorflow)$ python3 SVM.py <source file> [-n | --components n_comp] [-c | --classification c_class] [-s | --split s_number]
```

>*options:*
>-   *source file:*  _.tsv_ Features of the Images dataset in a certain layer, belonging to a pre-trained model ex. `VGG16_5.tsv`
>-   *n_comp:*  Number of components for the PCA descomposition, ex.`-n 75`, default `50`
>-   *c_class:*  Decision function shape for the _C-Support Vector Classification_ as a parameter in `sklearn.svm.SVC()`, these are the following options (ex.`-c SVC_ovr`, default `SVC_ovo`)
>     -   *SVC_ovo: one-vs-one*
>     -   *SVC_ovr: one-vs-rest*
>-   *s_number:*  Number with which the database is divided into training and test sets (size of training set), ex.`-s 0.5`, default `0.7`

>*output:* 
>-   Display in a terminal screen the Confusion Matrix for the supervised learning algorithm SVM.
>-   Display in a terminal screen the Average Accuracy for the supervised learning algorithm SVM.
