# DataSplit.R

# Setup ####

library(haven)
library(dplyr)

# Split merged data into separate sets for each outcome variable

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