# DataCleaning.R

# Setup ####

library(haven)
library(dplyr)
library(stringr)
library(sjlabelled)
library(tidyr)
library(janitor)

overwriteXPT <- F # change to TRUE to overwrite existing merged data file

# Loading data ####

Nutrition_D1 <- read_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1TOT_L.xpt') 
Nutrition_D2 <- read_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR2TOT_L.xpt')
Medical_Conditions <- read_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.xpt')


# Cleaning and merging ####

#Keeping SEQN and columns Energy - Caffeine
Filtered_Nutrition_D1 <- Nutrition_D1 %>% select(SEQN, 33:75)
Filtered_Nutrition_D2 <- Nutrition_D2 %>% select(SEQN, 16:58)

#Rename the columns
colnames(Filtered_Nutrition_D1)[-1] <- get_label(Filtered_Nutrition_D1)[-1]
colnames(Filtered_Nutrition_D2)[-1] <- get_label(Filtered_Nutrition_D2)[-1]

#Remove brackets
colnames(Filtered_Nutrition_D1) <- gsub("\\s*\\(.*?\\)", "", colnames(Filtered_Nutrition_D1))
colnames(Filtered_Nutrition_D2) <- gsub("\\s*\\(.*?\\)", "", colnames(Filtered_Nutrition_D2))

#Combine the datasets by averaging them
Filtered_Nutrition_Avg <- Filtered_Nutrition_D1 %>%
  mutate(across(2:ncol(.), ~ (Filtered_Nutrition_D1[[cur_column()]] + Filtered_Nutrition_D2[[cur_column()]]) / 2))

#Remove NA values
Filtered_Nutrition_Avg_Cleaned <- na.omit(Filtered_Nutrition_Avg)
na_rows_removed <- nrow(Filtered_Nutrition_Avg) - nrow(Filtered_Nutrition_Avg_Cleaned)

#Select the columns we want and rename them
Filtered_Medical_Conditions <- Medical_Conditions %>%
  select(
    SEQN, MCQ010, AGQ030, MCQ160A, MCQ160B, MCQ160C, MCQ160E, MCQ160M, MCQ220
  ) %>%
  rename(
    Asthma = MCQ010,
    Hay_Fever = AGQ030,
    Arthritis = MCQ160A,
    Congestive_Heart_Failure = MCQ160B,
    Coronary_Heart_Disease = MCQ160C,
    Heart_Attack = MCQ160E,
    Thyroid_Problems = MCQ160M,
    Cancer = MCQ220
  )

# Replace all NA values with 0 in both datasets
Filtered_Medical_Conditions[is.na(Filtered_Medical_Conditions)] <- 0

# Merge both datasets by SEQN, keeping only matching samples
Merged_Data <- inner_join(Filtered_Nutrition_Avg_Cleaned, Filtered_Medical_Conditions, by = "SEQN")


# Write to file ####

# write to XPT (have to remove spaces in names and shorten a couple first)
if (overwriteXPT){
write_xpt(clean_names(Merged_Data,replace=c(unsaturated='unsat'))
          ,path='Merged_Data.xpt')
}
