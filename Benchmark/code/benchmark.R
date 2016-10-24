library(mlr)
library(batchtools)
library(plyr)
library(OpenML)

dir = "/home/probst/Abschlussarbeiten/BoostedRF/MA/Benchmark"
setwd(paste0(dir,"/results"))
source(paste0(dir,"/code/benchmark_defs.R"))

unlink("benchmark-rf-brf", recursive = TRUE)
regis = makeExperimentRegistry("benchmark-rf-brf", 
                               packages = c("mlr", "OpenML", "methods"), 
                               source = paste0(dir, "/code/benchmark_defs.R"),
                               work.dir = paste0(dir, "/results"),
                               conf.file = paste0(dir,"/code/.batchtools.conf.R")
)
regis$cluster.functions = makeClusterFunctionsMulticore(ncpus = 10) 

# add selected OML datasets as problems
for (did in OMLDATASETS) {
  data = list(did = did)
  addProblem(name = as.character(did), data = data)
}

# add one generic 'algo' that evals the RF in hyperpar space
addAlgorithm("eval", fun = function(job, data, instance,  ...) {
  par.vals = list(...)
  oml.dset = getOMLDataSet(data$did)             
  task = convertOMLDataSetToMlr(oml.dset)
  type = getTaskType(task)
  # set here better defaults for each package?
  if(type == "classif") {
    learners = list(makeLearner("classif.randomForestSRC", par.vals = list(ntree = 5000), predict.type = "prob"), makeLearner("classif.brf.conv", predict.type = "prob"))
    }
  if(type == "regr"){
    learners = list(makeLearner("regr.randomForestSRC", par.vals = list(ntree = 5000)), makeLearner("regr.brf.conv"))
  }
  measures = MEASURES(type)
  rdesc = makeResampleDesc("RepCV", folds = 2, reps = 2, stratify = FALSE)
  configureMlr(on.learner.error = "warn", show.learner.output = FALSE)
  bmr = benchmark(learners, task, rdesc, measures, keep.pred = FALSE, models = FALSE, show.info = TRUE)
  bmr
})

set.seed(124)
ades = data.frame(c(1))
addExperiments(algo.designs = list(eval = ades))
summarizeExperiments()

ids = chunkIds(findNotDone(), chunk.size = 5)
submitJobs(ids)
# ranger braucht ca. 3 h für regression + classification
submitJobs(5)

#submitJobs(ids, resources = list(chunk.ncpus = 9))
getStatus()
getErrorMessages()

res_classif_load = reduceResultsList(ids = 5, fun = function(r) as.list(r), reg = regis)

regis = loadRegistry("benchmark-rf-brf")
# zu Debugzwecken
#lrn.id = "ranger"
#par.vals = as.list(ades[1,-1])
#data$did = OMLDATASETS[13]
