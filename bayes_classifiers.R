# Setup ####

library(tidyverse)
library(tidymodels)
library(discrim)
library(knitr)
library(haven)


# Data-wrangling ####

nutr <- read_xpt('Merged_Data.xpt')

nutr <- nutr |> 
  mutate(across(45:52, ~factor(.x, 
                               levels=c(1,2), 
                               labels=c('Yes','No'))))

nutrFilt <- list()

nutrNames <- names(nutr)

for (j in 45:52){
  nutrFilt[[j-44]] <- nutr |> 
    dplyr::select(!(45:52), j)
}

names(nutrFilt) <- names(nutr)[45:52]

nutrFilt$asthma <- nutrFilt$asthma |> 
  filter(asthma %in% c('Yes','No'))

nutrFilt$hay_fever <- nutrFilt$hay_fever |> 
  filter(hay_fever %in% c('Yes','No'))

nutrFilt$arthritis <- nutrFilt$arthritis |> 
  filter(arthritis %in% c('Yes','No'))

nutrFilt$congestive_heart_failure <- nutrFilt$congestive_heart_failure |> 
  filter(congestive_heart_failure %in% c('Yes','No'))

nutrFilt$coronary_heart_disease <- nutrFilt$coronary_heart_disease |> 
  filter(coronary_heart_disease %in% c('Yes','No'))

nutrFilt$heart_attack <- nutrFilt$heart_attack |> 
  filter(heart_attack %in% c('Yes','No'))

nutrFilt$thyroid_problems <- nutrFilt$thyroid_problems |> 
  filter(thyroid_problems %in% c('Yes','No'))

nutrFilt$cancer <- nutrFilt$cancer |> 
  filter(cancer %in% c('Yes','No'))


# Model training ####

resampleAccuracies <- function(model,recs,folds){
  nutrWorks <- list()
  nutrFits <- list()
  nutrAccuracies <- list()
  
  for (w in 1:length(recs)){
    
    nutrWorks[[w]] <- workflow() |> 
      add_model(model) |> 
      add_recipe(recs[[w]])
    
    nutrFits[[w]] <- fit_resamples(nutrWorks[[w]],
                                   folds[[w]],
                                   metrics=metric_set(accuracy,roc_auc),
                                   control = control_resamples(save_pred=T))
    
    nutrAccuracies[[w]] <- collect_metrics(nutrFits[[w]]) |> 
      dplyr::select(!c(.estimator,.config,n)) |> 
      mutate(outcome=names(nutrFilt)[w],.before=1)
    
  }
  
  names(nutrAccuracies) <- names(nutrFits) <- names(recs)
  
  fits <- nutrFits
  
  accuracies <- bind_rows(nutrAccuracies)
  
  return(list(accuracies=accuracies,fits=fits))
}


## Outliers included ####

nutrRecs <- list(
  recipe(asthma ~ ., data=nutrFilt$asthma),
  recipe(hay_fever ~ ., data=nutrFilt$hay_fever),
  recipe(arthritis ~ ., data=nutrFilt$arthritis),
  recipe(congestive_heart_failure ~ ., data=nutrFilt$congestive_heart_failure),
  recipe(coronary_heart_disease ~ ., data=nutrFilt$coronary_heart_disease),
  recipe(heart_attack ~ ., data=nutrFilt$heart_attack),
  recipe(thyroid_problems ~ ., data=nutrFilt$thyroid_problems),
  recipe(cancer ~ ., data=nutrFilt$cancer)
)

nutrFolds <- list()

for (w in 1:length(nutrFilt)){
  nutrRecs[[w]] <- nutrRecs[[w]] |> 
    update_role(seqn,new_role='ID')
  
  nutrFolds[[w]] <- vfold_cv(nutrFilt[[w]],v=10)
}

names(nutrRecs) <- names(nutrFolds) <- names(nutrFilt)


### Naive Bayes ####

nbSpec <- naive_Bayes() |>
  set_mode('classification') |>
  set_engine('naivebayes') |>
  set_args(usekernel=F)

resamples <- resampleAccuracies(model=nbSpec,recs=nutrRecs,folds=nutrFolds)

nbFits <- resamples$fits
nbAccuracies <- resamples$accuracies


### Linear discriminant analysis ####

ldaSpec <- discrim_linear() |> 
  set_mode('classification') |>
  set_engine('MASS')

resamples <- resampleAccuracies(model=ldaSpec,recs=nutrRecs,folds=nutrFolds)

ldaFits <- resamples$fits
ldaAccuracies <- resamples$accuracies


### Quadratic discriminant analysis ####

qdaSpec <- discrim_quad() |> 
  set_mode('classification') |>
  set_engine('MASS')

resamples <- resampleAccuracies(model=qdaSpec,recs=nutrRecs,folds=nutrFolds)

qdaFits <- resamples$fits
qdaAccuracies <- resamples$accuracies


## Outliers removed ####

nutrCropped <- nutr |> 
  filter(!if_any(1:44, rstatix::is_outlier))

nutrFilt <- list()

nutrNames <- names(nutrCropped)

for (j in 45:52){
  nutrFilt[[j-44]] <- nutrCropped |> 
    dplyr::select(!(45:52), j)
}

names(nutrFilt) <- names(nutrCropped)[45:52]

nutrFilt$asthma <- nutrFilt$asthma |> 
  filter(asthma %in% c('Yes','No'))

nutrFilt$hay_fever <- nutrFilt$hay_fever |> 
  filter(hay_fever %in% c('Yes','No'))

nutrFilt$arthritis <- nutrFilt$arthritis |> 
  filter(arthritis %in% c('Yes','No'))

nutrFilt$congestive_heart_failure <- nutrFilt$congestive_heart_failure |> 
  filter(congestive_heart_failure %in% c('Yes','No'))

nutrFilt$coronary_heart_disease <- nutrFilt$coronary_heart_disease |> 
  filter(coronary_heart_disease %in% c('Yes','No'))

nutrFilt$heart_attack <- nutrFilt$heart_attack |> 
  filter(heart_attack %in% c('Yes','No'))

nutrFilt$thyroid_problems <- nutrFilt$thyroid_problems |> 
  filter(thyroid_problems %in% c('Yes','No'))

nutrFilt$cancer <- nutrFilt$cancer |> 
  filter(cancer %in% c('Yes','No'))

for (w in 1:length(nutrFilt)){
  nutrFilt[[w]] <- nutrFilt[[w]] |> 
    dplyr::select(!where(is.numeric), where(~ is.numeric(.x) && var(.x)!=0))
}

nutrRecs <- list(
  recipe(asthma ~ ., data=nutrFilt$asthma),
  recipe(hay_fever ~ ., data=nutrFilt$hay_fever),
  recipe(arthritis ~ ., data=nutrFilt$arthritis),
  recipe(congestive_heart_failure ~ ., data=nutrFilt$congestive_heart_failure),
  recipe(coronary_heart_disease ~ ., data=nutrFilt$coronary_heart_disease),
  recipe(heart_attack ~ ., data=nutrFilt$heart_attack),
  recipe(thyroid_problems ~ ., data=nutrFilt$thyroid_problems),
  recipe(cancer ~ ., data=nutrFilt$cancer)
)


nutrFolds <- list()

for (w in 1:length(nutrFilt)){
  nutrRecs[[w]] <- nutrRecs[[w]] |> 
    update_role(seqn,new_role='ID')
  
  nutrFolds[[w]] <- vfold_cv(nutrFilt[[w]],v=10)
}

names(nutrRecs) <- names(nutrFolds) <- names(nutrFilt)

### Naive Bayes ####

resamples <- resampleAccuracies(model=nbSpec,recs=nutrRecs,folds=nutrFolds)

nbFits_outliers <- resamples$fits
nbAccuracies_outliers <- resamples$accuracies


### Linear discriminant analysis ####

resamples <- resampleAccuracies(model=ldaSpec,recs=nutrRecs,folds=nutrFolds)

ldaFits_outliers <- resamples$fits
ldaAccuracies_outliers <- resamples$accuracies


### Quadratic discriminant analysis ####

resamples <- resampleAccuracies(model=qdaSpec,recs=nutrRecs,folds=nutrFolds)

qdaFits_outliers <- resamples$fits
qdaAccuracies_outliers <- resamples$accuracies


# Plotting ####

## Outliers included ####

### Accuracy plots ####

classifiers <- c('Naive Bayes', 'LDA', 'QDA')

accuracies <- list(nb=nbAccuracies,
                   lda=ldaAccuracies,
                   qda=qdaAccuracies)

for (t in 1:length(classifiers)){
  accuracies[[t]] <- accuracies[[t]] |> 
    mutate(Classifier=classifiers[t],.before=1)
}

ROC_AUCs <- bind_rows(accuracies) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  filter(.metric=='roc_auc') |> 
  mutate(outcome=str_replace_all(outcome,'_',' '))

accuracies <- bind_rows(accuracies) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  filter(.metric=='accuracy') |> 
  mutate(outcome=str_replace_all(outcome,'_',' ')) |> 
  rowwise() |> 
  mutate(mean=mean*100,std_err=std_err*100)

bayesPlot <- accuracies |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  scale_y_continuous(limits=c(0,100))+
  labs(y='Prediction accuracy (%)',x='Diagnosed with...',title='Diagnosis prediction accuracy of Bayes\' classfiers')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

# bayesPlot


### ROC(-AUC) plots ####

nbROCs <- list()
for (w in 1:length(nbFits)){
  nbROCs[[w]] <- collect_predictions(nbFits[[w]]) |> 
    roc_curve(truth=names(nbFits)[w],.pred_Yes) |> 
    mutate(outcome=names(nbFits[w]),.before=1)
}
nbROCs <- bind_rows(nbROCs) |> 
  mutate(Classifier='Naive Bayes')

ldaROCs <- list()
for (w in 1:length(ldaFits)){
  ldaROCs[[w]] <- collect_predictions(ldaFits[[w]]) |> 
    roc_curve(truth=names(ldaFits)[w],.pred_Yes) |> 
    mutate(outcome=names(ldaFits[w]),.before=1)
}
ldaROCs <- bind_rows(ldaROCs) |> 
  mutate(Classifier='LDA')

qdaROCs <- list()
for (w in 1:length(qdaFits)){
  qdaROCs[[w]] <- collect_predictions(qdaFits[[w]]) |> 
    roc_curve(truth=names(qdaFits)[w],.pred_Yes) |> 
    mutate(outcome=names(qdaFits[w]),.before=1)
}
qdaROCs <- bind_rows(qdaROCs) |> 
  mutate(Classifier='QDA')

ROCs <- bind_rows(list(nbROCs,ldaROCs,qdaROCs)) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  mutate(outcome=str_replace_all(outcome,'_',' '))

bayesROCPlot <- ROCs |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~Classifier)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',name="Diagnosed with...")+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of Bayes\' classifiers')

# bayesROCPlot


ldaROCPlot <- ROCs |> 
  filter(Classifier=='LDA') |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~outcome)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',guide = 'none')+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of LDA classifier')

# ldaROCPlot


bayesROC_AUCPlot <- ROC_AUCs |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_hline(yintercept=0.5,lty=2,lwd=1.5)+
  scale_y_continuous(limits=c(0,1))+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  labs(y='ROC area under the curve',x='Diagnosed with...',title='Areas under the ROC curves for diagnosis predictions of Bayes\' classfiers')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

# bayesROC_AUCPlot




## Outliers removed ####

### Accuracy plots ####

accuracies_outliers <- list(nb=nbAccuracies_outliers,
                   lda=ldaAccuracies_outliers,
                   qda=qdaAccuracies_outliers)

for (t in 1:length(classifiers)){
  accuracies_outliers[[t]] <- accuracies_outliers[[t]] |> 
    mutate(Classifier=classifiers[t],.before=1)
}

ROC_AUCs_outliers <- bind_rows(accuracies_outliers) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  filter(.metric=='roc_auc') |> 
  mutate(outcome=str_replace_all(outcome,'_',' '))

accuracies_outliers <- bind_rows(accuracies_outliers) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  filter(.metric=='accuracy') |> 
  mutate(outcome=str_replace_all(outcome,'_',' ')) |> 
  rowwise() |> 
  mutate(mean=mean*100,std_err=std_err*100)

bayesPlot_outliers <- accuracies_outliers |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  scale_y_continuous(limits=c(0,100))+
  labs(y='Prediction accuracy (%)',x='Diagnosed with...',title='Diagnosis prediction accuracy of Bayes\' classfiers – no outliers')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

# bayesPlot_outliers


### ROC(-AUC) plots ####

nbROCs_outliers <- list()
for (w in 1:length(nbFits_outliers)){
  nbROCs_outliers[[w]] <- collect_predictions(nbFits_outliers[[w]]) |> 
    roc_curve(truth=names(nbFits_outliers)[w],.pred_Yes) |> 
    mutate(outcome=names(nbFits_outliers[w]),.before=1)
}
nbROCs_outliers <- bind_rows(nbROCs_outliers) |> 
  mutate(Classifier='Naive Bayes')

ldaROCs_outliers <- list()
for (w in 1:length(ldaFits_outliers)){
  ldaROCs_outliers[[w]] <- collect_predictions(ldaFits_outliers[[w]]) |> 
    roc_curve(truth=names(ldaFits_outliers)[w],.pred_Yes) |> 
    mutate(outcome=names(ldaFits_outliers[w]),.before=1)
}
ldaROCs_outliers <- bind_rows(ldaROCs_outliers) |> 
  mutate(Classifier='LDA')

qdaROCs_outliers <- list()
for (w in 1:length(qdaFits_outliers)){
  qdaROCs_outliers[[w]] <- collect_predictions(qdaFits_outliers[[w]]) |> 
    roc_curve(truth=names(qdaFits_outliers)[w],.pred_Yes) |> 
    mutate(outcome=names(qdaFits_outliers[w]),.before=1)
}
qdaROCs_outliers <- bind_rows(qdaROCs_outliers) |> 
  mutate(Classifier='QDA')

ROCs_outliers <- bind_rows(list(nbROCs_outliers,ldaROCs_outliers,qdaROCs_outliers)) |> 
  mutate(Classifier=factor(Classifier,levels=c('Naive Bayes','LDA','QDA'))) |> 
  mutate(outcome=str_replace_all(outcome,'_',' '))

bayesROCPlot_outliers <- ROCs_outliers |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~Classifier)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',name="Diagnosed with...")+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of Bayes\' classifiers – no outliers')

# bayesROCPlot_outliers


ldaROCPlot_outliers <- ROCs_outliers |> 
  filter(Classifier=='LDA') |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~outcome)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',guide = 'none')+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of LDA classifier – no outliers')

# ldaROCPlot_outliers


bayesROC_AUCPlot_outliers <- ROC_AUCs_outliers |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_hline(yintercept=0.5,lty=2,lwd=1.5)+
  scale_y_continuous(limits=c(0,1))+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  labs(y='ROC area under the curve',x='Diagnosed with...',title='Areas under the ROC curves for diagnosis predictions of Bayes\' classfiers – no outliers')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

# bayesROC_AUCPlot_outliers




# Writing plots to file ####

# NOTE: UN-COMMENT THE FOLLOWING TO SAVE THE PLOTS TO A NEW FOLDER IN THE WORKING DIRECTORY, WITH PROPER SIZES AND RESOLUTIONS


# figs <- c('bayesPlot',
#           'bayesPlot_outliers',
#           'bayesROC_AUCPlot',
#           'bayesROC_AUCPlot_outliers',
#           'bayesROCPlot',
#           'bayesROCPlot_outliers',
#           'ldaROCPlot',
#           'ldaROCPlot_outliers')
# 
# fignames <- c('bayes classifiers accuracy comparison',
#               'bayes classifiers accuracy comparison - no outliers',
#               'bayes classifiers ROC-AUC comparison',
#               'bayes classifiers ROC-AUC comparison - no outliers',
#               'bayes classifiers ROC curves comparison',
#               'bayes classifiers ROC curves comparison - no outliers',
#               'LDA classifier ROC curves comparison',
#               'LDA classifier ROC curves comparison - no outliers')
# 
# folderpath <- paste(getwd(),'/figures/', sep='')
# dir.create(folderpath)
# 
# widths <- c(rep(2200,4),rep(4400*.75,2), rep(1800,2))
# heights <- c(rep(1440,4), rep(1440*.75,2),rep(1600,2))
# ress <- c(rep(250,8))
# 
# for (i in 1:length(figs)) {
#   fig <- figs[i]
#   png(filename=paste(folderpath, fignames[i], '.png', sep=''), width=widths[i], height=heights[i], res=ress[i])
#   print(get(fig))
#   dev.off()
# }

