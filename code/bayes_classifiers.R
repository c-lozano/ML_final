# Setup ####

library(tidyverse)
library(tidymodels)
library(discrim)
library(knitr)

plotsToWindow <- F # change to T to show plots in RStudio
plotsToFile <- T # change to T to overwrite the PNGs in ../figures/


# Data-wrangling ####

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source('DataSplit.R')


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
  labs(y='Prediction accuracy (%)',x='Diagnosed with...',title='Diagnosis prediction accuracy of Bayes\' classifiers', subtitle='With 10-fold cross-validation')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

if(plotsToWindow) bayesPlot


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

if(plotsToWindow) bayesROCPlot


ldaROCPlot <- ROCs |> 
  filter(Classifier=='LDA') |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~outcome)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',guide = 'none')+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of LDA classifier')

if(plotsToWindow) ldaROCPlot


bayesROC_AUCPlot <- ROC_AUCs |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_hline(yintercept=0.5,lty=2,lwd=1.5)+
  scale_y_continuous(limits=c(0,1))+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  labs(y='ROC area under the curve',x='Diagnosed with...',title='Areas under the ROC curves for diagnosis predictions of Bayes\' classifiers', subtitle='With 10-fold cross-validation')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

if(plotsToWindow) bayesROC_AUCPlot




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
  labs(y='Prediction accuracy (%)',x='Diagnosed with...',title='Diagnosis prediction accuracy of Bayes\' classifiers – no outliers', subtitle='With 10-fold cross-validation')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

if(plotsToWindow) bayesPlot_outliers


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

if(plotsToWindow) bayesROCPlot_outliers


ldaROCPlot_outliers <- ROCs_outliers |> 
  filter(Classifier=='LDA') |> 
  ggplot(aes(x=1-specificity, y=sensitivity, color=outcome))+
  facet_wrap(~outcome)+
  geom_path()+
  geom_abline(lty=2,lwd=1.5)+
  scale_color_brewer(palette='Dark2',guide = 'none')+
  labs(x = 'False positive rate', y='True positive rate', title='Diagnosis prediction ROC curves of LDA classifier – no outliers')

if(plotsToWindow) ldaROCPlot_outliers


bayesROC_AUCPlot_outliers <- ROC_AUCs_outliers |> 
  ggplot(aes(outcome,mean,fill=outcome))+
  facet_wrap(~Classifier)+
  geom_col(position='dodge')+
  geom_hline(yintercept=0.5,lty=2,lwd=1.5)+
  scale_y_continuous(limits=c(0,1))+
  geom_errorbar(aes(ymin = mean-std_err, ymax = mean+std_err), position = "dodge")+
  scale_fill_brewer(guide='none',palette = 'Dark2')+
  labs(y='ROC area under the curve',x='Diagnosed with...',title='Areas under the ROC curves for diagnosis predictions of Bayes\' classifiers – no outliers', subtitle='With 10-fold cross-validation')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

if(plotsToWindow) bayesROC_AUCPlot_outliers




# Writing plots to file ####

if (plotsToFile){
  figs <- c('bayesPlot',
            'bayesPlot_outliers',
            'bayesROC_AUCPlot',
            'bayesROC_AUCPlot_outliers',
            'bayesROCPlot',
            'bayesROCPlot_outliers',
            'ldaROCPlot',
            'ldaROCPlot_outliers')
  
  fignames <- c('bayes_classifiers_accuracy_comparison',
                'bayes_classifiers_accuracy_comparison_no_outliers',
                'bayes_classifiers_ROC-AUC_comparison',
                'bayes_classifiers_ROC-AUC_comparison_no_outliers',
                'bayes_classifiers_ROC_curves_comparison',
                'bayes_classifiers_ROC_curves_comparison_no_outliers',
                'LDA_classifier_ROC_curves_comparison',
                'LDA_classifier_ROC_curves_comparison_no_outliers')
  
  folderpath <- paste(getwd(),'/figures/bayes/', sep='')
  dir.create(folderpath)
  
  widths <- c(rep(2200,4),rep(4400*.75,2), rep(1800,2))
  heights <- c(rep(1440,4), rep(1440*.75,2),rep(1600,2))
  ress <- c(rep(250,8))
  
  for (i in 1:length(figs)) {
    fig <- figs[i]
    png(filename=paste(folderpath, fignames[i], '.png', sep=''), width=widths[i], height=heights[i], res=ress[i])
    print(get(fig))
    dev.off()
  }
}

# Panda ####

#                               _,add8ba,
#                             ,d888888888b,
#                            d8888888888888b                        _,ad8ba,_
#                           d888888888888888)                     ,d888888888b,
#                           I8888888888888888 _________          ,8888888888888b
#                  __________`Y88888888888888P"""""""""""baaa,__ ,888888888888888,
#             ,adP"""""""""""9888888888P""^                 ^""Y8888888888888888I
#          ,a8"^           ,d888P"888P^                           ^"Y8888888888P'
#        ,a8^            ,d8888'                                     ^Y8888888P'
#       a88'           ,d8888P'                                        I88P"^
#     ,d88'           d88888P'                                          "b,
#    ,d88'           d888888'                                            `b,
#   ,d88'           d888888I                                              `b,
#   d88I           ,8888888'            ___                                `b,
#  ,888'           d8888888          ,d88888b,              ____            `b,
#  d888           ,8888888I         d88888888b,           ,d8888b,           `b
# ,8888           I8888888I        d8888888888I          ,88888888b           8,
# I8888           88888888b       d88888888888'          8888888888b          8I
# d8886           888888888       Y888888888P'           Y8888888888,        ,8b
# 88888b          I88888888b      `Y8888888^             `Y888888888I        d88,
# Y88888b         `888888888b,      `""""^                `Y8888888P'       d888I
# `888888b         88888888888b,                           `Y8888P^        d88888
#  Y888888b       ,8888888888888ba,_          _______        `""^        ,d888888
#  I8888888b,    ,888888888888888888ba,_     d88888888b               ,ad8888888I
#  `888888888b,  I8888888888888888888888b,    ^"Y888P"^      ____.,ad88888888888I
#   88888888888b,`888888888888888888888888b,     ""      ad888888888888888888888'
#   8888888888888698888888888888888888888888b_,ad88ba,_,d88888888888888888888888
#   88888888888888888888888888888888888888888b,`"""^ d8888888888888888888888888I
#   8888888888888888888888888888888888888888888baaad888888888888888888888888888'
#   Y8888888888888888888888888888888888888888888888888888888888888888888888888P
#   I888888888888888888888888888888888888888888888P^  ^Y8888888888888888888888'
#   `Y88888888888888888P88888888888888888888888888'     ^88888888888888888888I
#    `Y8888888888888888 `8888888888888888888888888       8888888888888888888P'
#     `Y888888888888888  `888888888888888888888888,     ,888888888888888888P'
#      `Y88888888888888b  `88888888888888888888888I     I888888888888888888'
#        "Y8888888888888b  `8888888888888888888888I     I88888888888888888'
#          "Y88888888888P   `888888888888888888888b     d8888888888888888'
#             ^""""""""^     `Y88888888888888888888,    888888888888888P'
#                              "8888888888888888888b,   Y888888888888P^
#                               `Y888888888888888888b   `Y8888888P"^
#                                 "Y8888888888888888P     `""""^
#                                   `"YY88888888888P'
#                                        ^""""""""'
