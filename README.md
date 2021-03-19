# LongitudinalAMDNet

This repository contains the Pytorch implementation for:

**LongitudinalAMDNet: A Temporal Correlation Structure Guided Deep Learning Framework for Predicting Longitudinal Age-Related Macular Degeneration Severity**<br/>Alireza Ganjdanesh, Jipeng Zhang, Anand Swaroop, Emily Chew, Ying Ding, Wei Chen&dagger;, Heng Huang&dagger;



<div align="left">
  <img src=./Figures/LongitudinalPred.png width = '293px' height = '183px'>
</div>



**Abstract:** Age-related  Macular  Degeneration  (AMD)  is  the  principal  cause  of  blindness  in  developed countries, and  its  prevalence  will  increase to  288 million  people  in  2040.  Therefore,  automated grading and prediction methods can be highly beneficial for recognizing susceptible subjects to late-AMD  by  enabling  clinicians to  start  preventive  actions  for  them.  Clinically,  AMD  severity grade  is  quantified  by  Color  Fundus  Photographs  (CFP)  of  the  retina,  and  deep  learning  based methods so far have focused on either grading AMD severity or future late-AMD risk prediction based  on  the  current  CFP. In  this  paper, we aim  to  develop  a  unified multi-class  classification model which can tackle both grading the current CFP and predicting the longitudinal outcome, i.e., whether the subject will be in late-AMD in a future time point. Furthermore, we develop a new temporal-correlation-structure guided Generative Adversarial Network (GAN) model that learns the complex patterns of temporal changes in CFPs of the subjects in consecutive time points, which enables us to perform longitudinal prediction forCFPs corresponding to the classifier’s outcome, and  by  doing  so,  provides interpretability for the classifier’s decisions for  ophthalmologists by forecasting the changes in AMD symptoms in the future CFPs. We used about 30,000 CFP images from 4628 participants in the Age-Related Eye Disease Study (AREDS), the largest longitudinal dataset available for AMD. Our classifier showed averaged 0.905 (95% confidence interval 0.886 -0.922) Area Under the Curve (AUC) and 0.762 (95% confidence interval 0.733 –0.792) accuracy on 3 class classification problem of simultaneously grading current time point’s AMD condition and predicting late AMD progression of the subject in the future time point. We further validated our model on the UK Biobank dataset, where our model showed average 0.905 accuracy and 0.797 sensitivity in grading 300 chosen CFP images.


