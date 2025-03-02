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
                                   metrics=metric_set(accuracy,roc_auc))
    
    nutrAccuracies[[w]] <- collect_metrics(nutrFits[[w]]) |> 
      dplyr::select(!c(.estimator,.config,n)) |> 
      mutate(outcome=names(nutrFilt)[w],.before=1)
    
  }
  
  names(nutrAccuracies) <- names(recs)
  
  accuracies <- bind_rows(nutrAccuracies)
  
  return(accuracies)
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


### Naive Bayes ####

nbSpec <- naive_Bayes() |>
  set_mode('classification') |>
  set_engine('naivebayes') |>
  set_args(usekernel=F)

nbAccuracies <- resampleAccuracies(model=nbSpec,recs=nutrRecs,folds=nutrFolds)


### Linear discriminant analysis ####

ldaSpec <- discrim_linear() |> 
  set_mode('classification') |>
  set_engine('MASS')

ldaAccuracies <- resampleAccuracies(model=ldaSpec,recs=nutrRecs,folds=nutrFolds)


### Quadratic discriminant analysis ####

qdaSpec <- discrim_quad() |> 
  set_mode('classification') |>
  set_engine('MASS')

qdaAccuracies <- resampleAccuracies(model=qdaSpec,recs=nutrRecs,folds=nutrFolds)


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

### Naive Bayes ####

nbAccuracies_outliers <- resampleAccuracies(model=nbSpec,recs=nutrRecs,folds=nutrFolds)


### Linear discriminant analysis ####

ldaAccuracies_outliers <- resampleAccuracies(model=ldaSpec,recs=nutrRecs,folds=nutrFolds)


### Quadratic discriminant analysis ####

qdaAccuracies_outliers <- resampleAccuracies(model=qdaSpec,recs=nutrRecs,folds=nutrFolds)


# Plotting ####

## Outliers included ####

classifiers <- c('Naive Bayes', 'LDA', 'QDA')

accuracies <- list(nb=nbAccuracies,
                   lda=ldaAccuracies,
                   qda=qdaAccuracies)

for (t in 1:length(classifiers)){
  accuracies[[t]] <- accuracies[[t]] |> 
    mutate(Classifier=classifiers[t],.before=1)
}
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


## Outliers removed ####

accuracies_outliers <- list(nb=nbAccuracies_outliers,
                   lda=ldaAccuracies_outliers,
                   qda=qdaAccuracies_outliers)

for (t in 1:length(classifiers)){
  accuracies_outliers[[t]] <- accuracies_outliers[[t]] |> 
    mutate(Classifier=classifiers[t],.before=1)
}
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
  labs(y='Prediction accuracy (%)',x='Diagnosed with...',title='Diagnosis prediction accuracy of Bayes\' classfiers – outliers removed')+
  theme(axis.text.x=element_text(angle=45, vjust = 1, hjust=1))

# bayesPlot_outliers


# Writing plots to file ####
# NOTE: UN-COMMENT THE FOLLOWING TO SAVE THE PLOTS TO A NEW FOLDER IN THE WORKING DIRECTORY, WITH PROPER SIZES AND RESOLUTIONS

# figs <- c('bayesPlot','bayesPlot_outliers')
# 
# fignames <- c('bayes classifiers accuracy comparison', 
#               'bayes classifiers accuracy comparison -outliers')
# 
# folderpath <- paste(getwd(),'/figures/', sep='')
# dir.create(folderpath)
# 
# for (i in 1:length(figs)) {
#   fig <- figs[i]
#   png(filename=paste(folderpath, fignames[i], '.png', sep=''), width=2200, height=1440, res=250)
#   print(get(fig))
#   dev.off()
# }

