
library(mlr)

makeRLearner.classif.brf.conv = function() {
  makeRLearnerClassif(
    cl = "classif.brf.conv",
    package = "MASS",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "forest.size", lower = 20, default = 300),
      #makeNumericLearnerParam(id = "init.weights", default = FALSE),
      makeNumericLearnerParam(id = "weight.treshold", default = 20, lower = 20),
      makeLogicalLearnerParam(id = "leaf.weights", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "converge", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "sample.weights", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "stoptreeOut", default = FALSE, tunable = FALSE),
      makeNumericLearnerParam(id = "smoothness", default = 200, lower = 20),
      makeNumericLearnerParam(id = "iqrfac", default = 1.5),
      makeNumericLearnerParam(id = "conv.treshold.reg", default = 1),
      makeNumericLearnerParam(id = "conv.treshold.clas", default = 0.001)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob"),
    name = "Boosted Random Forest",
    short.name = "brf.conv",
    note = ""
  )
}


trainLearner.classif.brf.conv = function(.learner, .task, .subset, .weights = NULL, ...) {
  f = getTaskTargetNames(.task)
  data = getTaskData(.task, subset = .subset)
  TY = data[, f]
  TX = data[,colnames(data) != f, drop = FALSE]
  brf.conv(TY = TY, TX = TX, ...)
}

predictLearner.classif.brf.conv = function(.learner, .model, .newdata, ...) {
  if (.learner$predict.type == "response") {
    p = brf.predict(.model$learner.model, data = .newdata, prob = FALSE) }
  else {
    p = brf.predict(.model$learner.model, data = .newdata, prob = TRUE)
  }
  return(p)
}

makeRLearner.regr.brf.conv = function() {
  makeRLearnerRegr(
    cl = "regr.brf.conv",
    package = "MASS",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "forest.size", lower = 20, default = 500),
      #makeNumericLearnerParam(id = "init.weights", default = FALSE),
      makeNumericLearnerParam(id = "weight.treshold", default = 20, lower = 20),
      makeLogicalLearnerParam(id = "leaf.weights", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "converge", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "sample.weights", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "stoptreeOut", default = FALSE, tunable = FALSE),
      makeNumericLearnerParam(id = "smoothness", default = 30, lower = 20),
      makeNumericLearnerParam(id = "iqrfac", default = 1.5),
      makeNumericLearnerParam(id = "conv.treshold.reg", default = 1),
      makeNumericLearnerParam(id = "conv.treshold.clas", default = 0.01)
    ),
    properties = c("numerics", "factors"),
    name = "Boosted Random Forest",
    short.name = "brf.conv",
    note = ""
  )
}

trainLearner.regr.brf.conv = function(.learner, .task, .subset, .weights = NULL, ...) {
  f = getTaskTargetNames(.task)
  data = getTaskData(.task, subset = .subset)
  TY = data[, f]
  TX = data[,colnames(data) != f, drop = FALSE]
  brf.conv(TY = TY, TX = TX, ...)
}

predictLearner.regr.brf.conv = function(.learner, .model, .newdata, ...) {
  brf.predict(.model$learner.model, data = .newdata, prob = FALSE)
}

data(iris)

# Iris Task
## Define the task
task = makeClassifTask(id = "tutorial", data = iris, target = "Species")
## Define the learner
lrn = makeLearner("classif.brf.conv")
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
lrn = makeLearner("classif.brf.conv", predict.type = "prob")
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

# Regression
## Define the task
task = bh.task
bh.data = getTaskData(bh.task)
## Define the learner
lrn = makeLearner("regr.brf.conv")
## Define the resampling strategy
rdesc = makeResampleDesc(method = "CV")
trn = train(lrn, task)
prd = predict(trn, newdata = bh.data)
## Do the resampling
r = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE)
r

