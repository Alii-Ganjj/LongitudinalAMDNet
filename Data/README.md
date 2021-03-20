## Data Preparation

### Age-Related Eye Disease Study (AREDS) Dataset
The National Eye Institute (NEI) Age-Related Eye Disease Study (AREDS) dataset is available at **[Database of Genotypes and Phenotypes (dbGap) phs000001.v3.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1)**. 

### UK Biobank Dataset
The UK Biobank (UKB) dataset is accessible through **[the online showcase of UK Biobank resources](https://biobank.ndph.ox.ac.uk/crystal/).**

## Data Preprocessing

We followed the same preprocessing protocol for [DeepSeeNet](https://www.sciencedirect.com/science/article/pii/S0161642018321857?casa_token=-DUY6w9R7wwAAAAA:q1iL-PXTXh7a_xVTZGXWYosxrQQnHXezan2Ow8E_ZFnNwB7ARLl7F9ryia_6b66V04yHU-Y0Cg) available in their [repository](https://github.com/ncbi-nlp/DeepSeeNet). Each color fundus image is cropped to a square which encompasses the Macula region, and then, the square image was resized to 224 * 224 pixels. Our code for preprocessing all the images in the dataset is available in `main_resize_images.py` file. As the images in AREDS dataset are from two batches 2010 and 2014, this code assumes that the images for batch 2010 (2014) are in the LongitudinalAMDNet/Datasets/AREDS/img_2010 (2014).

## Data Partitioning

To make reproducing the experiments easier, we added three files with the name format 'FullVisitsGapXThr10.pkl' (X = 4, 6, or 8) containing the images used for each subject in experiments with the pairs with 4 (2), 6 (3), and 8 (4) months (years) time gap. These files are used in [here](https://github.com/Alii-Ganjj/LongitudinalAMDNet/blob/9deb08bb3f91c1979d1a7ec5cdc615512d28464c/Data/AMDDataAREDS.py#L176) to partition the dataset into the train, validation, and test sets. The names of images used in each experiment can be found by running the code until [here](https://github.com/Alii-Ganjj/LongitudinalAMDNet/blob/9deb08bb3f91c1979d1a7ec5cdc615512d28464c/Data/AMDDataAREDS.py#L164).


