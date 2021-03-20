## Data Preparation

### Age-Related Eye Disease Study (AREDS) Dataset
The National Eye Institute (NEI) Age-Related Eye Disease Study (AREDS) dataset is available at **[Database of Genotypes and Phenotypes (dbGap) phs000001.v3.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1)**. 

### UK Biobank Dataset
The UK Biobank (UKB) dataset is accessible through **[the online showcase of UK Biobank resources](https://biobank.ndph.ox.ac.uk/crystal/).**

## Data Preprocessing

We followed the same preprocessing protocol for [DeepSeeNet](https://www.sciencedirect.com/science/article/pii/S0161642018321857?casa_token=-DUY6w9R7wwAAAAA:q1iL-PXTXh7a_xVTZGXWYosxrQQnHXezan2Ow8E_ZFnNwB7ARLl7F9ryia_6b66V04yHU-Y0Cg) available in their [repository](https://github.com/ncbi-nlp/DeepSeeNet). Each color fundus image is cropped to a square which encompasses the Macula region, and then, the square image was resized to 224 * 224 pixels.
