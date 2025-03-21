# Loading the necessary libraries and data set.
library(haven)
library(tidyverse)
library(tidymodels)
library(randomForest)
library(rpart.plot)
library(vip)
library(modeldata)
library(ggfortify)
library(ggplot2)
df <- read_xpt("Merged_Data.xpt")
df[45:52] <- lapply(df[45:52], as.factor) # Convert response variables into factors

# Keep only "yes" (1) and "no" (2) and remove "missing" or "don't know" for all responses
asthma <- df[1:45] |> filter(asthma != 9) |> mutate(asthma = droplevels(asthma))
hay_fever <- df[c(1:44,46)] |> filter(hay_fever != 9) |> mutate(hay_fever = droplevels(hay_fever))
arthritis <- df[c(1:44,47)] |> filter(!arthritis %in% c(0,9)) |> mutate(arthritis = droplevels(arthritis))
congestive <- df[c(1:44,48)] |> filter(!congestive_heart_failure %in% c(0,9)) |> mutate(congestive_heart_failure = droplevels(congestive_heart_failure))
coronary <- df[c(1:44,49)] |> filter(!coronary_heart_disease %in% c(0,9)) |> mutate(coronary_heart_disease = droplevels(coronary_heart_disease))
heart_attack <- df[c(1:44,50)] |> filter(!heart_attack %in% c(0,9)) |> mutate(heart_attack = droplevels(heart_attack))
thyroid <- df[c(1:44,51)] |> filter(!thyroid_problems %in% c(0,7,9)) |> mutate(thyroid_problems = droplevels(thyroid_problems))
cancer <- df[c(1:44,52)] |> filter(!cancer %in% c(0,7,9)) |> mutate(cancer = droplevels(cancer))

# Training a basic decision tree model
class_tree_spec <- decision_tree() |>
  set_engine('rpart') |>
  set_mode("classification")

## Asthma
split <- initial_split(asthma, prop = 1/2)
asthma_train <- training(split)
asthma_test <- testing(split)

asthma_recipe <- recipe(asthma ~ ., data = asthma_train) |>
  update_role('seqn', new_role = 'ID')
asthma_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(asthma_recipe) 
class_tree_fit <- asthma_workflow |> fit(asthma_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
asthma_class_pred <- augment(class_tree_fit, new_data = asthma_test)
asthma_train_accuracy <- augment(class_tree_fit, new_data = asthma_train) |> accuracy(truth = asthma, estimate = .pred_class)
asthma_test_accuracy <- asthma_class_pred |> accuracy(truth = asthma, estimate = .pred_class)
asthma_auc <- asthma_class_pred |> roc_auc(truth = asthma, .pred_2)
asthma_class_pred |> conf_mat(truth = asthma, estimate = .pred_class)

## Hay fever
split <- initial_split(hay_fever, prop = 1/2)
hay_fever_train <- training(split)
hay_fever_test <- testing(split)

hay_fever_recipe <- recipe(hay_fever ~ ., data = hay_fever_train) |>
  update_role('seqn', new_role = 'ID')
hay_fever_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(hay_fever_recipe) 
class_tree_fit <- hay_fever_workflow |> fit(hay_fever_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
hay_fever_class_pred <- augment(class_tree_fit, new_data = hay_fever_test)
hay_fever_train_accuracy <- augment(class_tree_fit, new_data = hay_fever_train) |> accuracy(truth = hay_fever, estimate = .pred_class)
hay_fever_test_accuracy <- hay_fever_class_pred |> accuracy(truth = hay_fever, estimate = .pred_class)
hay_fever_auc <- hay_fever_class_pred |> roc_auc(truth = hay_fever, .pred_2)
hay_fever_class_pred |> conf_mat(truth = hay_fever, estimate = .pred_class)

## Arthritis
split <- initial_split(arthritis, prop = 1/2)
arthritis_train <- training(split)
arthritis_test <- testing(split)

arthritis_recipe <- recipe(arthritis ~ ., data = arthritis_train) |>
  update_role('seqn', new_role = 'ID')
arthritis_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(arthritis_recipe) 
class_tree_fit <- arthritis_workflow |> fit(arthritis_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
arthritis_class_pred <- augment(class_tree_fit, new_data = arthritis_test)
arthritis_train_accuracy <- augment(class_tree_fit, new_data = arthritis_train)|> accuracy(truth = arthritis, estimate = .pred_class)
arthritis_test_accuracy <- arthritis_class_pred |> accuracy(truth = arthritis, estimate = .pred_class)
arthritis_auc <- arthritis_class_pred |> roc_auc(truth = arthritis, .pred_1)
arthritis_class_pred |> conf_mat(truth = arthritis, estimate = .pred_class)

## Congestive Heart Failure
split <- initial_split(congestive, prop = 1/2)
congestive_train <- training(split)
congestive_test <- testing(split)

congestive_recipe <- recipe(congestive_heart_failure ~ ., data = congestive_train) |>
  update_role('seqn', new_role = 'ID')
congestive_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(congestive_recipe) 
class_tree_fit <- congestive_workflow |> fit(congestive_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
congestive_class_pred <- augment(class_tree_fit, new_data = congestive_test)
congestive_train_accuracy <- augment(class_tree_fit, new_data = congestive_train)|> accuracy(truth = congestive_heart_failure, estimate = .pred_class)
congestive_test_accuracy <- congestive_class_pred |> accuracy(truth = congestive_heart_failure, estimate = .pred_class)
congestive_auc <- congestive_class_pred |> roc_auc(truth = congestive_heart_failure, .pred_1)
congestive_class_pred |> conf_mat(truth = congestive_heart_failure, estimate = .pred_class)

## Coronary Heart Disease
split <- initial_split(coronary, prop = 1/2)
coronary_train <- training(split)
coronary_test <- testing(split)

coronary_recipe <- recipe(coronary_heart_disease ~ ., data = coronary_train) |>
  update_role('seqn', new_role = 'ID')
coronary_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(coronary_recipe) 
class_tree_fit <- coronary_workflow |> fit(coronary_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
coronary_class_pred <- augment(class_tree_fit, new_data = coronary_test)
coronary_train_accuracy <- augment(class_tree_fit, new_data = coronary_train)|> accuracy(truth = coronary_heart_disease, estimate = .pred_class)
coronary_test_accuracy <- coronary_class_pred |> accuracy(truth = coronary_heart_disease, estimate = .pred_class)
coronary_auc <- coronary_class_pred |> roc_auc(truth = coronary_heart_disease, .pred_1)
coronary_class_pred |> conf_mat(truth = coronary_heart_disease, estimate = .pred_class)

## Heart Attack
split <- initial_split(heart_attack, prop = 1/2)
heart_attack_train <- training(split)
heart_attack_test <- testing(split)

heart_attack_recipe <- recipe(heart_attack ~ ., data = heart_attack_train) |>
  update_role('seqn', new_role = 'ID')
heart_attack_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(heart_attack_recipe) 
class_tree_fit <- heart_attack_workflow |> fit(heart_attack_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
heart_attack_class_pred <- augment(class_tree_fit, new_data = heart_attack_test)
heart_attack_train_accuracy <- augment(class_tree_fit, new_data = heart_attack_train)|> accuracy(truth = heart_attack, estimate = .pred_class)
heart_attack_test_accuracy <- heart_attack_class_pred |> accuracy(truth = heart_attack, estimate = .pred_class)
heart_attack_auc <- heart_attack_class_pred |> roc_auc(truth = heart_attack, .pred_1)
heart_attack_class_pred |> conf_mat(truth = heart_attack, estimate = .pred_class)

## Thyroid
split <- initial_split(thyroid, prop = 1/2)
thyroid_train <- training(split)
thyroid_test <- testing(split)

thyroid_recipe <- recipe(thyroid_problems ~ ., data = thyroid_train) |>
  update_role('seqn', new_role = 'ID')
thyroid_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(thyroid_recipe) 
class_tree_fit <- thyroid_workflow |> fit(thyroid_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
thyroid_class_pred <- augment(class_tree_fit, new_data = thyroid_test)
thyroid_train_accuracy <- augment(class_tree_fit, new_data = thyroid_train)|> accuracy(truth = thyroid_problems, estimate = .pred_class)
thyroid_test_accuracy <- thyroid_class_pred |> accuracy(truth = thyroid_problems, estimate = .pred_class)
thyroid_auc <- thyroid_class_pred |> roc_auc(truth = thyroid_problems, .pred_1)
thyroid_class_pred |> conf_mat(truth = thyroid_problems, estimate = .pred_class)

## Cancer
split <- initial_split(cancer, prop = 1/2)
cancer_train <- training(split)
cancer_test <- testing(split)

cancer_recipe <- recipe(cancer ~ ., data = cancer_train) |>
  update_role('seqn', new_role = 'ID')
cancer_workflow <- workflow() |>
  add_model(class_tree_spec) |>
  add_recipe(cancer_recipe) 
class_tree_fit <- cancer_workflow |> fit(cancer_train)
class_tree_fit |> extract_fit_engine() |> rpart.plot(roundint = F, box.palette = "Greys")
cancer_class_pred <- augment(class_tree_fit, new_data = cancer_test)
cancer_train_accuracy <- augment(class_tree_fit, new_data = cancer_train)|> accuracy(truth = cancer, estimate = .pred_class)
cancer_test_accuracy <- cancer_class_pred |> accuracy(truth = cancer, estimate = .pred_class)
cancer_auc <- cancer_class_pred |> roc_auc(truth = cancer, .pred_2)
cancer_class_pred |> conf_mat(truth = cancer, estimate = .pred_class)

## Plotting the accuracy and AUC
### Accuracy
accuracy_df <- data.frame(
  Model = rep(c('Asthma','Hay Fever','Arthritis','Congestive Heart Failure','Coronary Heart Disease','Heart Attack','Thyroid Problems','Cancer'), each = 2),
  Accuracy = c(asthma_train_accuracy$.estimate, asthma_test_accuracy$.estimate, hay_fever_train_accuracy$.estimate, hay_fever_test_accuracy$.estimate, 
               arthritis_train_accuracy$.estimate, arthritis_test_accuracy$.estimate, congestive_train_accuracy$.estimate, congestive_test_accuracy$.estimate,
               coronary_train_accuracy$.estimate, coronary_test_accuracy$.estimate, heart_attack_train_accuracy$.estimate, heart_attack_test_accuracy$.estimate,
               thyroid_train_accuracy$.estimate, thyroid_test_accuracy$.estimate, cancer_train_accuracy$.estimate, cancer_test_accuracy$.estimate),
  Set = rep(c('Training','Testing'), times = 8)
)

ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Set)) +
  geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "Training vs Testing Accuracy for Each Model",
       x = "Model",
       y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### AUC
auc_df <- data.frame(
  Model = c('Asthma','Hay Fever','Arthritis','Congestive Heart Failure','Coronary Heart Disease','Heart Attack','Thyroid Problems','Cancer'),
  AUC = c(asthma_auc$.estimate, hay_fever_auc$.estimate, arthritis_auc$.estimate, congestive_auc$.estimate,
          coronary_auc$.estimate, heart_attack_auc$.estimate, thyroid_auc$.estimate, cancer_auc$.estimate)
)

ggplot(auc_df, aes(x = Model, y = AUC)) +
  geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "AUC for Each Model",
       x = "Model",
       y = "AUC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0,1) +
  geom_hline(yintercept = 0.5, linetype = 'dashed', color = 'red', linewidth = 1.2)

# Train a random forest model
bagging_spec <- rand_forest() |>
  set_engine("randomForest", importance = T) |>
  set_mode("classification")

## Asthma
asthma_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(asthma_recipe)

asthma_forest_fit <- asthma_workflow |> fit(asthma_train)
asthma_forest_train_accuracy <- augment(asthma_forest_fit, new_data = asthma_train) |> 
  accuracy(truth = asthma, estimate = .pred_class)
asthma_forest_pred <- augment(asthma_forest_fit, new_data = asthma_test)
asthma_forest_test_accuracy <- asthma_class_pred |> accuracy(truth = asthma, estimate = .pred_class)
asthma_forest_auc <- asthma_forest_pred |> roc_auc(truth = asthma, .pred_1)
asthma_class_pred |> conf_mat(truth = asthma, estimate = .pred_class)

## Hay Fever
hay_fever_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(hay_fever_recipe)

hay_fever_forest_fit <- hay_fever_workflow |> fit(hay_fever_train)
hay_fever_forest_train_accuracy <- augment(hay_fever_forest_fit, new_data = hay_fever_train) |> 
  accuracy(truth = hay_fever, estimate = .pred_class)
hay_fever_forest_pred <- augment(hay_fever_forest_fit, new_data = hay_fever_test)
hay_fever_forest_test_accuracy <- hay_fever_class_pred |> accuracy(truth = hay_fever, estimate = .pred_class)
hay_fever_forest_auc <- hay_fever_forest_pred |> roc_auc(truth = hay_fever, .pred_1)
hay_fever_class_pred |> conf_mat(truth = hay_fever, estimate = .pred_class)

## Arthritis
arthritis_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(arthritis_recipe)

arthritis_forest_fit <- arthritis_workflow |> fit(arthritis_train)
arthritis_forest_train_accuracy <- augment(arthritis_forest_fit, new_data = arthritis_train) |> 
  accuracy(truth = arthritis, estimate = .pred_class)
arthritis_forest_pred <- augment(arthritis_forest_fit, new_data = arthritis_test)
arthritis_forest_test_accuracy <- arthritis_class_pred |> accuracy(truth = arthritis, estimate = .pred_class)
arthritis_forest_auc <- arthritis_forest_pred |> roc_auc(truth = arthritis, .pred_1)
arthritis_class_pred |> conf_mat(truth = arthritis, estimate = .pred_class)

## Congestive Heart Failure
congestive_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(congestive_recipe)

congestive_forest_fit <- congestive_workflow |> fit(congestive_train)
congestive_forest_train_accuracy <- augment(congestive_forest_fit, new_data = congestive_train) |> 
  accuracy(truth = congestive_heart_failure, estimate = .pred_class)
congestive_forest_pred <- augment(congestive_forest_fit, new_data = congestive_test)
congestive_forest_test_accuracy <- congestive_class_pred |> accuracy(truth = congestive_heart_failure, estimate = .pred_class)
congestive_forest_auc <- congestive_forest_pred |> roc_auc(truth = congestive_heart_failure, .pred_1)
congestive_class_pred |> conf_mat(truth = congestive_heart_failure, estimate = .pred_class)

## Coronary Heart Disease
coronary_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(coronary_recipe)

coronary_forest_fit <- coronary_workflow |> fit(coronary_train)
coronary_forest_train_accuracy <- augment(coronary_forest_fit, new_data = coronary_train) |> 
  accuracy(truth = coronary_heart_disease, estimate = .pred_class)
coronary_forest_pred <- augment(coronary_forest_fit, new_data = coronary_test)
coronary_forest_test_accuracy <- coronary_class_pred |> accuracy(truth = coronary_heart_disease, estimate = .pred_class)
coronary_forest_auc <- coronary_forest_pred |> roc_auc(truth = coronary_heart_disease, .pred_1)
coronary_class_pred |> conf_mat(truth = coronary_heart_disease, estimate = .pred_class)

## Heart Attack
heart_attack_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(heart_attack_recipe)

heart_attack_forest_fit <- heart_attack_workflow |> fit(heart_attack_train)
heart_attack_forest_train_accuracy <- augment(heart_attack_forest_fit, new_data = heart_attack_train) |> 
  accuracy(truth = heart_attack, estimate = .pred_class)
heart_attack_forest_pred <- augment(heart_attack_forest_fit, new_data = heart_attack_test)
heart_attack_forest_test_accuracy <- heart_attack_class_pred |> accuracy(truth = heart_attack, estimate = .pred_class)
heart_attack_forest_auc <- heart_attack_forest_pred |> roc_auc(truth = heart_attack, .pred_1)
heart_attack_class_pred |> conf_mat(truth = heart_attack, estimate = .pred_class)

## Thyroid
thyroid_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(thyroid_recipe)

thyroid_forest_fit <- thyroid_workflow |> fit(thyroid_train)
thyroid_forest_train_accuracy <- augment(thyroid_forest_fit, new_data = thyroid_train) |> 
  accuracy(truth = thyroid_problems, estimate = .pred_class)
thyroid_forest_pred <- augment(thyroid_forest_fit, new_data = thyroid_test)
thyroid_forest_test_accuracy <- thyroid_class_pred |> accuracy(truth = thyroid_problems, estimate = .pred_class)
thyroid_forest_auc <- thyroid_forest_pred |> roc_auc(truth = thyroid_problems, .pred_1)
thyroid_class_pred |> conf_mat(truth = thyroid_problems, estimate = .pred_class)

## Cancer
cancer_workflow <- workflow() |>
  add_model(bagging_spec) |>
  add_recipe(cancer_recipe)

cancer_forest_fit <- cancer_workflow |> fit(cancer_train)
cancer_forest_train_accuracy <- augment(cancer_forest_fit, new_data = cancer_train) |> 
  accuracy(truth = cancer, estimate = .pred_class)
cancer_forest_pred <- augment(cancer_forest_fit, new_data = cancer_test)
cancer_forest_test_accuracy <- cancer_class_pred |> accuracy(truth = cancer, estimate = .pred_class)
cancer_forest_auc <- cancer_forest_pred |> roc_auc(truth = cancer, .pred_1)
cancer_class_pred |> conf_mat(truth = cancer, estimate = .pred_class)

## Plotting the accuracy and AUC
### Accuracy
forest_accuracy_df <- data.frame(
  Model = rep(c('Asthma','Hay Fever','Arthritis','Congestive Heart Failure','Coronary Heart Disease','Heart Attack','Thyroid Problems','Cancer'), each = 2),
  Accuracy = c(asthma_forest_train_accuracy$.estimate, asthma_forest_test_accuracy$.estimate, hay_fever_forest_train_accuracy$.estimate, hay_fever_forest_test_accuracy$.estimate, 
               arthritis_forest_train_accuracy$.estimate, arthritis_forest_test_accuracy$.estimate, congestive_forest_train_accuracy$.estimate, congestive_forest_test_accuracy$.estimate,
               coronary_forest_train_accuracy$.estimate, coronary_forest_test_accuracy$.estimate, heart_attack_forest_train_accuracy$.estimate, heart_attack_forest_test_accuracy$.estimate,
               thyroid_forest_train_accuracy$.estimate, thyroid_forest_test_accuracy$.estimate, cancer_forest_train_accuracy$.estimate, cancer_forest_test_accuracy$.estimate),
  Set = rep(c('Training','Testing'), times = 8)
)

ggplot(forest_accuracy_df, aes(x = Model, y = Accuracy, fill = Set)) +
  geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "Training vs Testing Accuracy for Each Model",
       x = "Model",
       y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### AUC
forest_auc_df <- data.frame(
  Model = c('Asthma','Hay Fever','Arthritis','Congestive Heart Failure','Coronary Heart Disease','Heart Attack','Thyroid Problems','Cancer'),
  AUC = c(arthritis_auc$.estimate, hay_fever_auc$.estimate, arthritis_auc$.estimate, congestive_auc$.estimate,
          coronary_auc$.estimate, heart_attack_auc$.estimate, thyroid_auc$.estimate, cancer_auc$.estimate)
)

ggplot(forest_auc_df, aes(x = Model, y = AUC)) +
  geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "AUC for Each Model",
       x = "Model",
       y = "AUC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0,1) +
  geom_hline(yintercept = 0.5, linetype = 'dashed', color = 'red', linewidth = 1.2)

## Tuning
set.seed(123)
cv_folds <- vfold_cv(arthritis_train, v = 10)

rf_model <- rand_forest(mtry = tune()) |>
  set_engine("randomForest", importance = TRUE) |>
  set_mode("classification")

rf_recipe <- recipe(arthritis ~ ., data = arthritis_train)

rf_workflow <- workflow() |>
  add_model(rf_model) |>
  add_recipe(rf_recipe)

rf_grid <- grid_regular(
  mtry(range = c(1, 43)),
  levels = 43
)

rf_results <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc)
)

tuning_metrics <- rf_results |>
  collect_metrics() |>
  arrange(desc(mean))
tuning_metrics$mtry <- factor(tuning_metrics$mtry, levels = tuning_metrics$mtry)

best_mtry <- rf_results |> select_best(metric = "roc_auc")

final_rf_workflow <- finalize_workflow(rf_workflow, best_mtry)

final_rf_model <- fit(final_rf_workflow, data = arthritis_train)

final_rf_fit <- extract_fit_parsnip(final_rf_model)

arthritis_class_pred <- augment(final_rf_model, new_data = arthritis_test)
arthritis_class_pred |> accuracy(truth = arthritis, estimate = .pred_class)
arthritis_class_pred |> roc_auc(truth = arthritis, .pred_1)
arthritis_class_pred |> conf_mat(truth = arthritis, estimate = .pred_class)
vip(final_rf_fit)

ggplot(tuning_metrics, aes(x = mtry, y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err)) +
  labs(title = 'Random Forest Tuning Results', x = 'mtry', y = 'AUC') +
  theme_minimal() +
  ylim(0.5,0.7)
