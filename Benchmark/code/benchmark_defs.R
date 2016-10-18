load("/home/probst/Benchmarking/benchmark-mlr-openml/results/clas.RData")
load("/home/probst/Benchmarking/benchmark-mlr-openml/results/reg.RData")
tasks = rbind(clas_small, reg_small)

OMLDATASETS = tasks$did[!(tasks$did %in% c(1054, 1071, 1065))] # Cannot guess task.type from data! for these 3

MEASURES = function(x) switch(x, "classif" = list(acc, ber, mmce, multiclass.au1u, multiclass.brier, logloss, timetrain), "regr" = list(mse, mae, medae, medse, rsq, timetrain))

library(ranger)

brf.conv <- function(TY, TX, forest.size = 300, leaf.weights = FALSE, sample.weights = TRUE, init.weights = NULL, weight.threshold = 20,
                     smoothness = 200, conv.threshold.clas = 0.001, conv.threshold.reg = 1, converge = TRUE, iqrfac = 1.5) {
  
  TS <- as.data.frame(cbind(TY,TX))
  N <- dim(TX)[1]
  M <- dim(TX)[2]
  
  
  # classification or regression?
  classification <- is.factor(TY)
  regression <- is.numeric(TY) || is.integer(TY)
  if(!(classification || regression))
    stop(paste("Target class", class(TY), "is not suitable for brf"))
  
  if(regression && leaf.weights){
    stop("leaf.weights not possible if regression, set to FALSE")
  }
  
  if(!converge){
    L <- forest.size
    prediction.matrix <- matrix(data = 0, nrow = N, ncol = L)
    if(classification){
      weight.matrix <- matrix(data = 0, nrow = N, ncol = L)
    }
  }
  
  if(converge){
    s <- smoothness
    weight.matrix <- matrix(data = 0, nrow = N, ncol = 1)
    if(classification){
      t <- conv.threshold.clas
      oob_err <- vector(mode = "numeric")
      weighted.oob_err <- vector(mode = "numeric")
    }
    if(regression){
      t <- conv.threshold.reg
      mse.conv <- vector(mode = "numeric")
    }
    prediction.matrix <- matrix(data = 0, nrow = N, ncol = 1)
  }
  
  if(classification){
    prediction.correct <- vector(mode = "numeric", length = N)
  }
  
  oob.vector <- vector(mode = "numeric", length = N)
  weights <- vector(mode = "numeric", length = N)
  vote.final <- vector(mode = "numeric", length = N)
  weighted.vote.matrix <- matrix(data = 0, ncol = length(levels(TY)), nrow = N)
  weighted.vote.final <- vector(mode = "numeric", length = N)
  vote.matrix <- matrix(data = 0, ncol = length(levels(TY)), nrow = N)
  result <- list()
  forest <- list()
  
  if(regression){
    iqr <- IQR(TY, type = 7)
    qntl.75 <- quantile(TY)[4]
    qntl.25 <- quantile(TY)[2]
    bias <- vector(mode = "numeric", length = N)
  }
  
  # Levels nummerieren
  if(classification){
    result$lev.names <- levels(TY)
    lev.names <- levels(TY)
    levels(TY) <- c(1:length(levels(TY)))
  }
  
  # initialise weights for Training Instances of first step
  if(is.null(init.weights)){ # default is 1/N
    init.weights <- rep(1/N, N)
  }
  weights <- init.weights
  
  
  
  ##################### without convergence #####################
  if(!converge){
    for(l in 1:L){
      
      # build tree of sampled instances
      if(sample.weights){
        tr <- ranger(TY~., data = TS, num.trees = 1, case.weights = weights, write.forest = TRUE) 
      } else tr <- ranger(TY~., data = TS, num.trees = 1, case.weights = NULL, write.forest = TRUE) # only one tree is built, evtl mtry aus N(sqrt(M),M/50) 
      
      forest[[l]] <- tr
      
      # predictions of current tree, to be saved in data.frame
      prediction.matrix[,l] <- tr$predictions
      
      #save weights
      weight.matrix[,l] <- weights
      
      if(classification){
        oob.vector[which(!is.na(prediction.matrix[,l]))] <- oob.vector[which(!is.na(prediction.matrix[,l]))] + 1
        
        for(j in 1:length(levels(TY))){
          vote.matrix[,j] <- vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l]))
        }
        if(leaf.weights){
          for(j in 1:length(levels(TY))){
            weighted.vote.matrix[,j] <- weighted.vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l])) * weights
          }
        }
        
        vote.final <- levels(TY)[max.col(vote.matrix, ties.method="random")]
        if(leaf.weights){
          weighted.vote.final <- levels(TY)[max.col(weighted.vote.matrix, ties.method="random")]
        }
        
        prediction.correct[which(prediction.matrix[,l] == TY)] <- prediction.correct[which(prediction.matrix[,l] == TY)] + 1
      }
      if(regression){
        oob.vector[which(prediction.matrix[,l]!=0)] <- oob.vector[which(prediction.matrix[,l]!=0)] + 1
        vote.final <- apply(prediction.matrix,1,sum)/oob.vector
        vote.final[which(is.na(vote.final))] <- 0
        bias[which(!is.na(vote.final))] <- TY[which(!is.na(vote.final))] - vote.final[which(!is.na(vote.final))]
        var <- vector(mode = "numeric", length = N)
        for(j in 1:l){
          oob <- which(prediction.matrix[,j]!=0)
          var[oob] <- ((prediction.matrix[oob,j] - vote.final[oob])^2) + var[oob]
        }
        var <- var/oob.vector
        var[which(is.na(var))] <- 0
        mse <- var + (bias)^2
      }
      
      # weights: # trees where x is OOB and which predict correct value / # trees where x is OOB
      # only if already more than 20 trees in forest
      if(l > weight.threshold){
        if(classification){
          weights[which(!is.na(prediction.matrix[,l]))] <- 1 - (prediction.correct[which(!is.na(prediction.matrix[,l]))]/oob.vector[which(!is.na(prediction.matrix[,l]))])
        }
        else if(regression){
          # Ausreißer sollen nicht stärker gewichtet werden
          weights[which((TY < (qntl.75 + iqrfac*iqr)) == (TY > (qntl.25 - iqrfac*iqr)))] <- mse[which((TY < (qntl.75 + iqrfac*iqr)) == (TY > (qntl.25 - iqrfac*iqr)))]
        }
        # Normalisierung der Gewichte, sodass Summe = 1
        Z <- sum(weights)
        weights <- weights/Z
      }
    }
    
    # prediction of the forest in case of classification
    if(classification){
      result$oob.error <- sum(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
      result$type <- "classification"
      result$oob.error <- sum(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
      if(leaf.weights){
        result$weighted.oob.error <- sum(weighted.vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
        result$weights <- weight.matrix
      }
      result$leaf.weights <- leaf.weights
    }
    
    else if(regression){
      result$mse <- mean(mse, na.rm = T)
      result$type <- "regression"
    }
    result$forest <- forest
    result$num.trees <- forest.size
    result$num.levels <- length(levels(TY))
    result$levels <- levels(TY)
    class(result) <- "brf"
    
    return(result)
  }
  
  ########################## with convergence #############################
  
  if(converge){
    l <- 1
    convergence <- FALSE
    while(!convergence){
      
      # build tree of sampled instances with weights 
      if(sample.weights){
        tr <- ranger(TY~., data = TS, num.trees = 1, case.weights = weights, write.forest = TRUE) 
      } else tr <- ranger(TY~., data = TS, num.trees = 1, case.weights = NULL, write.forest = TRUE) # only one tree is built, evtl mtry aus N(sqrt(M),M/50) 
      
      # save tree in forest
      forest[[l]] <- tr
      
      # predictions of current tree, to be saved in data.frame
      if(l > 1){
        prediction.matrix <- cbind(prediction.matrix, tr$predictions)
        weight.matrix <- cbind(weight.matrix, weights)
      } else{
        prediction.matrix[,l] <- tr$predictions
        weight.matrix[,l] <- weights
      }
      
      if(classification){
        oob.vector[which(!is.na(prediction.matrix[,l]))] <- oob.vector[which(!is.na(prediction.matrix[,l]))] + 1 # wie oft war Beobachtung n OOB
        
        for(j in 1:length(levels(TY))){
          vote.matrix[,j] = vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l]))
        }
        if(leaf.weights){
          for(j in 1:length(levels(TY))){
            weighted.vote.matrix[,j] <- weighted.vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l])) * weights
          }
        }
        
        vote.final <- levels(TY)[max.col(vote.matrix, ties.method="random")]
        if(leaf.weights){
          weighted.vote.final <- levels(TY)[max.col(weighted.vote.matrix, ties.method="random")]
        }
      }
      
      if(regression){
        oob.vector[which(prediction.matrix[,l]!=0)] <- oob.vector[which(prediction.matrix[,l]!=0)] + 1
        vote.final <- apply(prediction.matrix,1,sum)/oob.vector
        vote.final[which(is.na(vote.final))] <- 0
        bias[which(oob.vector!=0)] <- TY[which(oob.vector!=0)] - vote.final[which(oob.vector!=0)]
        var <- vector(mode = "numeric", length = N)
        for(j in 1:l){
          oob <- which(prediction.matrix[,j]!=0) # all instances that are out of bag
          var[oob] <- ((prediction.matrix[oob,j] - vote.final[oob])^2) + var[oob] # varianz
        }
        var <- var/oob.vector
        var[which(is.na(var))] <- 0
        mse <- var + (bias)^2
      }
      
      if(classification){
        # how many times has instance n been predicted correctly
        prediction.correct[which(prediction.matrix[,l] == TY)] <- prediction.correct[which(prediction.matrix[,l] == TY)] + 1
        oob_err[l] <- sum(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
        if(leaf.weights){
          weighted.oob_err[l] <- sum(weighted.vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
        }
        if(l > s){
          if(!leaf.weights){
            moob <- mean(oob_err[c((l-s):l)])  # Mittelwert des Fehlers der letzten s Bäume
            diffoob <- vector(mode = "numeric")  
            diffoob <- abs(oob_err[c((l-s):l)] - moob)  # absolute Abweichung der letzten s Bäume vom Mittelwert
            mdiffoob <- mean(diffoob)  # Mittelwert der Abweichung
            
            # convergence?
            if(abs(mdiffoob) < t){  # which value of t would be a good one?
              convergence <- TRUE
            }
          } else {moob <- mean(weighted.oob_err[c((l-s):l)])  # Mittelwert des Fehlers der letzten s Bäume
          diffoob <- vector(mode = "numeric")  
          diffoob <- abs(weighted.oob_err[c((l-s):l)] - moob)  # absolute Abweichung der letzten s Bäume vom Mittelwert
          mdiffoob <- mean(diffoob)  # Mittelwert der Abweichung
          
          # convergence?
          if(abs(mdiffoob) < t){  # which value of t would be a good one?
            convergence <- TRUE
          }
          }
        } 
      }
      if(regression){
        mse.conv[l] <- mean(mse, na.rm = TRUE)
        if(l > s){
          mmse <- mean(mse.conv[c((l-s):l)]) # Mittel des MSEs der letzten s Bäume 
          diffmse <- abs(mse.conv[c((l-s):l)] - mmse)
          mdiffmse <- mean(diffmse)
          if(abs(mdiffmse) < t){
            convergence <- TRUE
          }
        }
      }
      # weights: # trees where x is OOB and which predict correct value / # trees where x is OOB
      # for reliability concerns, only if already more than 20 trees in forest
      if((l > weight.threshold) && !convergence){
        if(classification){
          weights[which(!is.na(prediction.matrix[,l]))] <- 1 - (prediction.correct[which(!is.na(prediction.matrix[,l]))]/oob.vector[which(!is.na(prediction.matrix[,l]))])
        }
        if(regression){
          # Ausreißer sollen nicht stärker gewichtet werden
          weights[which((TY < (qntl.75 + iqrfac*iqr)) == (TY > (qntl.25 - iqrfac*iqr)))] <- mse[which((TY < (qntl.75 + iqrfac*iqr)) == (TY > (qntl.25 - iqrfac*iqr)))]
        }
        # Normalisierung der Gewichte, sodass Summe = 1
        Z <- sum(weights)
        weights <- weights/Z
      }
      l <- l + 1
    }
    
    # Return Object
    result$num.trees <- l - 1
    
    if(classification){
      result$oob.error <- sum(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
      result$type <- "classification"
      if(leaf.weights){
        result$oob.error.weighted <- sum(weighted.vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])/N
        result$weights <- weight.matrix
      }
    }
    result$leaf.weights <- leaf.weights
    if(regression){
      result$mse <- mean(mse, na.rm = T)
      result$type <- "regression"
      result$mdiffmse <- mdiffmse
    }
    result$forest <- forest
    result$num.levels <- length(levels(TY))
    result$levels <- levels(TY)
    class(result) <- "brf"
    
    return(result) 
  }
}

brf.predict <- function(object, data, prob = FALSE){ 
  if(class(object)!="brf"){
    stop("object needs to be of class brf")
  }
  if(object$type == "regression" && prob){
    stop("probability only for classification possible")
  }
  predictions.matrix <- vector(mode="numeric")
  if(object$leaf.weights){
    vote.matrix <- matrix(data = 0, nrow = dim(data)[1], ncol = object$num.levels)
  }
  
  
  for(t in 1:(object$num.trees)){
    predictions.matrix <- cbind(predictions.matrix, predict(object$forest[[t]], data)$prediction)
    if(object$leaf.weights){
      for(i in 1:object$num.levels){
        vote.matrix[which(predictions.matrix[,t]==i),i] <- vote.matrix[which(predictions.matrix[,t]==i),i] + object$weights[which(predictions.matrix[,t]==i),t]
      }
    }
  }
  
  if (object$type == "classification") {
    if(!prob){
      if(!object$leaf.weights){
        predictions <- object$lev.names[as.numeric(apply(predictions.matrix, 1, function(x) names(which.max(table(x)))))]
        predictions <- factor(predictions, levels = object$lev.names)
      } 
      if(object$leaf.weights){
        predictions <- object$lev.names[as.numeric(apply(vote.matrix, 1, function(x) which.max(x)))]
        predictions <- factor(predictions, levels = object$lev.names)
      }
      
      return(predictions)
    }
    
    if(prob){
      predictions.prob <- vector("numeric")
      for(i in 1:object$num.levels){
        predictions.prob <- cbind(predictions.prob, apply(predictions.matrix,1, function(x) length(which(x==i))))
      }
      predictions.prob <- data.frame(predictions.prob/(object$num.trees))
      colnames(predictions.prob) <- object$lev.names
      return(as.matrix(predictions.prob))
    }
  }
  
  if (object$type == "regression") {
    return(apply(predictions.matrix, 1, mean))
  }
}


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