
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            ~~
##                          MODELLING BICEP CURL DATA                       ----
##                                                                            ~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            --
##---------------------------- LIBRARIES AND DATA-------------------------------
##                                                                            --
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(tidyverse)
library(magrittr)
library(caret)
library(lubridate)

training <- read_csv(
  "08_PracticalMachineLearning/CourseProject/pml-training.csv",
  na = c("", "NA", "#DIV/0!"),
  guess_max = Inf,
  show_col_types = FALSE
  ) %>% 
  mutate(
    cvtd_timestamp = dmy_hm(cvtd_timestamp),
    TimeMonth = month(cvtd_timestamp),
    TimeDay = day(cvtd_timestamp),
    DayOfWeek = wday(cvtd_timestamp),
    TimeHour = hour(cvtd_timestamp),
    TimeMinute = minute(cvtd_timestamp)
  ) %>% 
  select(
    -`...1`, -cvtd_timestamp
  )

testing <- read_csv(
  "08_PracticalMachineLearning/CourseProject/pml-testing.csv",
  na = c("", "NA", "#DIV/0!"),
  guess_max = Inf,
  show_col_types = FALSE
) %>% 
  mutate(
    cvtd_timestamp = dmy_hm(cvtd_timestamp),
    TimeMonth = month(cvtd_timestamp),
    TimeDay = day(cvtd_timestamp),
    DayOfWeek = wday(cvtd_timestamp),
    TimeHour = hour(cvtd_timestamp),
    TimeMinute = minute(cvtd_timestamp)
  ) %>% 
  select(
    -`...1`, -cvtd_timestamp
  )

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            --
##----------------------------------- EDA---------------------------------------
##                                                                            --
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#  The variables with all missing values in the testing set are the summary     
#  values for each window. Because the testing data are snapshots within a      
#  window (i.e. none of them are summary rows) I will not have summary data     
#  in the test set to generate predictions. I therefore exclude all of the      
#  rows which are completely empty in the testing set.                          

useless_columns <- testing %>% 
  select(
    where(~all(is.na(.)))
  ) %>% 
    colnames()

training %<>%
  select(-all_of(c(useless_columns, "new_window")))

testing %<>%
  select(-all_of(c(useless_columns, "new_window")))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                      Testing for Linear Separability                     ----
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#  I perform PCA on the data set and see if the classes are separable in the    
#  space spanned by the first and second components (and by what kernel).       

training_pca <- prcomp(
  x = training %>% select(-classe, -user_name),
  scale. = TRUE
)

training_pca_plot <- training_pca$x %>% 
  as_tibble() %>% 
  mutate(classe = training$classe) %>% 
  ggplot(
    aes(
      x = PC1,
      y = PC2,
      colour = classe
    )
  ) + 
  geom_point(alpha = 0.2)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            --
##--------------------------------- LDA/PLSDA-----------------------------------
##                                                                            --
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ctrl <- trainControl(
  method = "cv",
  number = 10,
  # repeats = 2,
  classProbs = TRUE,
  savePredictions = TRUE
)

ldaFit <- train(
  classe ~ .,
  data = training,
  method = "lda",
  preProc = c("center", "scale"),
  metric = "Accuracy",
  trControl = ctrl
)

plsFit <- train(
  classe ~ .,
  data = training,
  method = "pls", 
  tuneGrid = expand.grid(ncomp = 1:50),
  preProc = c("center", "scale"),
  metric = "Accuracy",
  trControl = ctrl
)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            --
##-------------------------------- NEURAL NET-----------------------------------
##                                                                            --
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nnFit <- train(
  classe ~ .,
  data = training,
  method = "nnet",
  preProc = c("center", "scale", "spatialSign"),
  tuneGrid = expand.grid(
    .size = 3:6, .decay = 0
  ),
  trace = FALSE,
  maxit = 2000,
  trControl = ctrl
)

nnBestTune <- data.frame(
  size = 6,
  decay = 0
)

write_rds(
  nnFit,
  "08_PracticalMachineLearning/CourseProject/nnFitCaretObject.rds"
)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##                                                                            --
##----------------------- OTHER NONLINEAR CLASSIFIERS---------------------------
##                                                                            --
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fdaFit <- train(
  classe ~ .,
  data = training,
  method = "fda",
  preProc = c("center", "scale"),
  tuneGrid = expand.grid(
    degree = 1:3,
    nprune = seq(2, 100, length.out = 10) %>% floor()
  ),
  trControl = ctrl
)

fdaBestTune <- data.frame(
  degree = 3,
  nprune = 78
)

library(kernlab)

sigmaRangeReduced <- training %>% 
  select(-classe, -user_name) %>% 
  as.matrix() %>% 
  sigest()

svmRGridReduced <- expand.grid(
  .sigma = sigmaRangeReduced[1],
  .C = 2^seq(-4,4)
)

svmRFit <- train(
  classe ~ ., 
  data = training,
  method = "svmRadial",
  metric = "Accuracy",
  preProc = c("center", "scale"),
  tuneGrid = svmRGridReduced,
  fit = FALSE,
  trControl = ctrl
)

svmRBestTune <- data.frame(
  sigma = 0.006195374,
  C = 16
)