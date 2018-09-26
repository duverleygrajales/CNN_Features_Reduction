# CNN_Features_Extractor
>   Bionformatics Project Polito

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
>-   Create a `file.tsv` with a _rootPATH_ and _label_ of each image, from the whole images dataset.

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

#### _SVM.py_

#### _KNN.py_
