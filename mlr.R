
library(mlr)
makeRLearner.classif.brf = function() {
  makeRLearnerClassif(
    cl = "classif.brf",
    package = "MASS",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "forest.size", lower = 20, default = 300),
      #makeNumericLearnerParam(id = "init.weights", default = FALSE),
      makeNumericLearnerParam(id = "weight.treshold", default = 20, lower = 20),
      makeLogicalLearnerParam(id = "leaf.weights", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "converge", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "sample.weights", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "stoptreeOut", default = FALSE, tunable = FALSE),
      makeNumericLearnerParam(id = "smoothness", default = 200, lower = 20),
      makeNumericLearnerParam(id = "conv.treshold.clas", default = 0.001)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob"),
    name = "Boosted Random Forest",
    short.name = "brf",
    note = ""
  )
}


trainLearner.classif.brf = function(.learner, .task, .subset, .weights = NULL, ...) {
  f = getTaskTargetNames(.task)
  data = getTaskData(.task, subset = .subset)
  TY = data[, f]
  TX = data[,colnames(data) != f, drop = FALSE]
  brf(TY = TY, TX = TX, ...)
}

predictLearner.classif.brf = function(.learner, .model, .newdata, ...) {
  if (.learner$predict.type == "response") {
    p = predict.brf(.model$learner.model, data = .newdata, prob = FALSE) }
  else {
    p = predict.brf(.model$learner.model, data = .newdata, prob = TRUE)
  }
  return(p)
}


data(iris)

# Iris Task
## Define the task
task = makeClassifTask(id = "tutorial", data = iris, target = "Species")
## Define the learner
lrn = makeLearner("classif.brf")
## Define the resampling strategy
rdesc = makeResampleDesc(method = "CV", stratify = TRUE)
trn = train(lrn, task)
prd = predict(trn, newdata = iris)
## Do the resampling
r = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE)
r

# Sonar Task
## Define the task
task = sonar.task
sonar = getTaskData(sonar.task) 
## Define the learner
lrn = makeLearner("classif.brf", predict.type = "prob")
trn = train(lrn, task)
prd = predict(trn, newdata = sonar)
performance(prd, measures = list(acc, ber, mmce, multiclass.au1u, multiclass.brier, logloss))

trn = train(makeLearner("classif.randomForest", predict.type = "prob"), task)
prd2 = predict(trn, newdata = sonar)
performance(prd2, measures = list(acc, ber, mmce, multiclass.au1u, multiclass.brier, logloss))

## Define the resampling strategy
rdesc = makeResampleDesc(method = "CV", stratify = TRUE)
## Do the resampling
r = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE, measures = measures)
r
r2 = resample(learner = makeLearner("classif.ranger", predict.type = "prob"), task = task, resampling = rdesc, show.info = FALSE, measures = measures)
r2
