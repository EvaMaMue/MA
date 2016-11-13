load("/home/probst/Benchmarking/benchmark-mlr-openml/results/clas.RData")
load("/home/probst/Benchmarking/benchmark-mlr-openml/results/reg.RData")
tasks = rbind(clas_small, reg_small)

OMLDATASETS = tasks$did[!(tasks$did %in% c(1054, 1071, 1065))] # Cannot guess task.type from data! for these 3

MEASURES = function(x) switch(x, "classif" = list(acc, ber, mmce, multiclass.au1u, multiclass.brier, logloss, timetrain), "regr" = list(mse, mae, medae, medse, rsq, timetrain))

library(randomForestSRC)

brf <- function(TY, TX, forest.size = 300, leaf.weights = TRUE, sample.weights = FALSE, init.weights = NULL, weight.threshold = 20,
                smoothness = 200, conv.threshold.clas = 0.001, converge = TRUE, stoptreeOut = F) {
  
  TS <- as.data.frame(cbind(TY,TX))
  N <- dim(TX)[1]
  M <- dim(TX)[2]
  
  
  # classification or regression?
  classification <- is.factor(TY)
  regression <- is.numeric(TY) || is.integer(TY)
  if(!classification)
    stop(paste("Target class", class(TY), "is not suitable for brf"))
  
  # Definition der nötigen Objekte
  
  if(!converge){ # Objekte, die nur nötig sind, falls Anzahl der Bäume a priori vorgegeben wird
    if(stoptreeOut){ # falls Ausgabe des Baumes gewünscht, nach welchem Algorithmus bei Konergenz abgebrochen hätte
      convergence <- FALSE
      s <- smoothness
      t <- conv.threshold.clas
    }
    L <- forest.size
    prediction.matrix <- matrix(data = 0, nrow = N, ncol = L) # Matrix[n,l], in der für jeden Baum l für jede Beobachtung n die Vorhersage abgespeichert wird
    weight.matrix <- matrix(data = 0, nrow = N, ncol = L) # Matrix[n,l]: Gewicht der Beobachtung n in Baum l
  }
  
  if(converge){ # Definition von Objekten, welche nur bei Nutzung des Konvergenzkriteriums nötig sind
    s <- smoothness # OOB Fehler soll über die letzten s Bäume "konstant" bleiben
    weight.matrix <- matrix(data = 0, nrow = N, ncol = 1) # Matrix[n,l]: Gewicht der Beobachtung n in Baum l
    t <- conv.threshold.clas # Schwellenwert, um welchen der Mittelwert des OOB Fehlers über die letzten s Bäume nicht abweichen soll !!!!
    prediction.matrix <- matrix(data = 0, nrow = N, ncol = 1) # Matrix[n,l] Vorhersage der Beobachtung n in Baum l
  }
  
  oob_err <- vector(mode = "numeric") # Vektor[l]: Out Of Bag Fehlerrate nach Fit den l-ten Baumes
  prediction.correct <- vector(mode = "numeric", length = N) # Vektor[n]: Anzahl Bäume korrekter Vorhersagen für n, falls Beobachtung n OOB
  oob.vector <- vector(mode = "numeric", length = N) # Vektor[n]: Anzahl der Bäume, für welche Beobachtung n OOB
  weights <- vector(mode = "numeric", length = N) # Vektor[n]: Gewicht für Beobachtung n (ändert sich in jedem Iterationsschritt)
  vote.matrix <- matrix(data = 0, ncol = length(levels(TY)), nrow = N) # Matrix[n,j]: Anzahl der Bäume mit Vorhersage Klasse j für Beobachtung n 
  vote.final <- vector(mode = "numeric", length = N) # Vektor[n]: majority Vote für Beobachtung n in aktueller Iteration (ändert sich in jedem Iterationsschritt)
  majClassNodesForest <- list() # Liste, die vorhergesagte Klasse für Beobachtung n angibt, je nachdem, in welchen Endknoten die Beobachtung fällt
  result <- list() # Liste, in denen Werte zur Ausgabe der Funktion abgespeichert werden
  forest <- list() # Liste, bestehend aus den Bäumen
  
  
  # Levels nummerieren
  
  result$lev.names <- levels(TY)
  lev.names <- levels(TY)
  levels(TY) <- c(1:length(levels(TY)))
  
  
  # initialise weights for Training Instances of first step
  if(is.null(init.weights)){ # default is 1/N
    init.weights <- rep(1/N, N)
  }
  weights <- init.weights
  
  
  
  # NO CONVERGENCE --------------------------------------------------------------------------------------------------------
  if(!converge){
    for(l in 1:L){
      
      # build tree of sampled instances
      if(sample.weights){
        tr <- rfsrc(TY~., data = TS, ntree = 1, case.wt = weights, forest = TRUE, membership = T) 
      } else tr <- rfsrc(TY~., data = TS, ntree = 1, case.wt = NULL, forest = TRUE, membership = T) 
      
      # save tree in forest
      forest[[l]] <- tr
      
      #save weights
      weight.matrix[,l] <- weights
      
      # predictions of current tree, to be saved in data.frame
      if(!leaf.weights){
        prediction.matrix[,l] <- tr$class.oob
        prediction.matrix[which(tr$inbag!=0),l] <- NA
      } 
      if(leaf.weights){
        #nenner <- table((as.factor(tr$membership))[which(tr$inbag != 0)]) # Anzahl Beobachtungen je Knoten
        
        classesNodes <- matrix(0, ncol = length(levels(TY)), nrow = tr$leaf.count) #Matrix[k,j]. Zeilen = Knoten k, Spalten = Klassen j, Zellwert = Summe der Gewichte aller Beobachtungen in Klasse j und Knoten k 
        sumweightsNodes <- list() # Summe der Gewichte in Klasse j in Knoten k: sumWeightsNodes[[j]][[k]][1]
        
        for(i in 1:length(levels(TY))){
          sumweightsNodes[[i]] <- lapply(1:(tr$leaf.count), 
                                         function(x) sum(weight.matrix[which(tr$membership == x & tr$inbag != 0 & TY == levels(TY)[i]),l]))
        }
        
        for(i in 1:length(levels(TY))){
          classesNodes[,i]<- as.vector(sapply(1:tr$leaf.count, function(x) sumweightsNodes[[i]][[x]][1]))
        }
        
        # Lege fest, welcher Klasse Beobachtung i zugeordnet wird, falls sie in Knoten k fällt -> majClassNodes[k] ist Klasse
        majClassNodes <- apply(classesNodes,1, function(x) which.max(x))
        
        # Abspeichern von majClassNodes für jeden Baum l
        majClassNodesForest[[l]] <- majClassNodes
        
        # Prediction der OOB Beobachtungen für Baum l, um OOB Fehlerrate berechnen zu können, falls inbag -> NA
        predictedClass <- apply(tr$membership,2, function(x) as.integer(majClassNodes[x]))
        predictedClass[which(tr$inbag!=0)] <- NA
        
        prediction.matrix[,l] <- predictedClass
      }
      
      
      oob.vector[which(!is.na(prediction.matrix[,l]))] <- oob.vector[which(!is.na(prediction.matrix[,l]))] + 1
      
      
      for(j in 1:length(levels(TY))){
        vote.matrix[,j] <- vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l]))
      }
      
      vote.final <- levels(TY)[max.col(vote.matrix, ties.method="random")]
      oob_err[l] <- mean(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])
      
      
      if(stoptreeOut){
        if(!convergence){
          if(l > s){
            moob <- mean(oob_err[c((l-s):l)])  # Mittelwert des Fehlers der letzten s Bäume
            diffoob <- vector(mode = "numeric")  
            diffoob <- abs(oob_err[c((l-s):l)] - moob)  # absolute Abweichung der letzten s Bäume vom Mittelwert
            mdiffoob <- mean(diffoob)  # Mittelwert der Abweichung
            
            # convergence?
            if(abs(mdiffoob) < t){  # which value of t would be a good one?
              stoptree <- l
              convergence <- TRUE
            }
          }
        }
      }
      prediction.correct[which(prediction.matrix[,l] == TY)] <- prediction.correct[which(prediction.matrix[,l] == TY)] + 1
      
      # weights: # trees where x is OOB and which predict correct value / # trees where x is OOB
      # only if already more than 20 trees in forest
      if(l > weight.threshold){
        weights[which(oob.vector!=0)] <- 1 - (prediction.correct[which(oob.vector!=0)]/oob.vector[which(oob.vector!=0)])
        
        # Normalisierung der Gewichte, sodass Summe = 1
        Z <- sum(weights)
        weights <- weights/Z
      }
    }
    
    # Speichern der Ausgabeobjekte der Funktion
    
    result$oob.error <- oob_err
    result$type <- "classification"
    result$leaf.weights <- leaf.weights
    result$predictions <- prediction.matrix
    result$forest <- forest
    result$num.trees <- forest.size
    result$num.levels <- length(levels(TY))
    result$levels <- levels(TY)
    class(result) <- "brf"
    if(leaf.weights){
      result$weights <- weight.matrix
      result$majClassNodesForest <- majClassNodesForest
    }
    if(stoptreeOut) result$stoptree <- stoptree
    
    return(result)
  }
  
  # CONVERGENCE ---------------------------------------------------------------------------------------------------
  
  if(converge){
    l <- 1
    convergence <- FALSE
    while(!convergence){
      
      # build tree of sampled instances
      if(sample.weights){
        tr <- rfsrc(TY~., data = TS, ntree = 1, case.wt = weights, forest = TRUE) 
      } else tr <- rfsrc(TY~., data = TS, ntree = 1, case.wt = NULL, forest = TRUE) # only one tree is built, evtl mtry aus N(sqrt(M),M/50) 
      
      # save tree in forest
      forest[[l]] <- tr
      
      
      
      # predictions of current tree, to be saved in data.frame
      if(l > 1){
        prediction.matrix <- cbind(prediction.matrix, tr$class.oob)
        prediction.matrix[which(tr$inbag==1),l] <- NA
        weight.matrix <- cbind(weight.matrix, weights)
      } else{
        prediction.matrix[,l] <- tr$class.oob
        prediction.matrix[which(tr$inbag==1),l] <- NA
        weight.matrix[,l] <- weights
      }
      
      if(leaf.weights){
        nenner <- table((as.factor(tr$membership))[which(tr$inbag != 0)]) # Anzahl Beobachtungen je Knoten
        
        #train$Species[which(a$membership == 1 & a$inbag == 0)]
        #weights[which(a$membership == 1 & a$inbag == 0)]
        classesNodes <- matrix(0, ncol = length(levels(TY)), nrow = tr$leaf.count) #Matrix[k,j]. Zeilen = Knoten k, Spalten = Klassen j, Zellwert = Summe der Gewichte aller Beobachtungen in Klasse j und Knoten k 
        sumweightsNodes <- list() # Summe der Gewichte in Klasse j in Knoten k: sumWeightsNodes[[j]][[k]][1]
        
        for(i in 1:length(levels(TY))){
          sumweightsNodes[[i]] <- lapply(1:(tr$leaf.count), 
                                         function(x) sum(weight.matrix[which(tr$membership == x & tr$inbag != 0 & TY == levels(TY)[i]),l]))
        }
        
        for(i in 1:length(levels(TY))){
          classesNodes[,i]<- as.vector(sapply(1:tr$leaf.count, function(x) sumweightsNodes[[i]][[x]][1]))
        }
        
        # Bestimme #Gewichte in Klasse j und Knoten k/#Beobachtungen in Knoten k -> Matrix[k,j]
        # anteilClassesNodes <- apply(classesNodes,2,function(x) x/nenner)
        # anteilClassesNodes[which(is.na(anteilClassesNodes))] <- 0
        
        # Lege fest, welcher Klasse Beobachtung i zugeordnet wird, falls sie in Knoten k fällt -> majClassNodes[k] ist Klasse
        majClassNodes <- apply(classesNodes,1, function(x) which.max(x))
        
        # Abspeichern von majClassNodes für jeden Baum l
        majClassNodesForest[[l]] <- majClassNodes
        
        # Prediction der OOB Beobachtungen für Baum l, um OOB Fehlerrate berechnen zu können, falls inbag -> NA
        
        predictedClass <- apply(tr$membership,2, function(x) as.integer(majClassNodes[x]))
        predictedClass[which(tr$inbag!=0)] <- NA
        
        if(l>1){prediction.matrix <- cbind(prediction.matrix, predictedClass)
        } else prediction.matrix[,l] <- predictedClass
      }
      
      oob.vector[which(!is.na(prediction.matrix[,l]))] <- oob.vector[which(!is.na(prediction.matrix[,l]))] + 1 # wie oft war Beobachtung n OOB
      
      for(j in 1:length(levels(TY))){
        vote.matrix[,j] = vote.matrix[,j] + as.numeric(prediction.matrix[,l]==levels(TY)[j] & !is.na(prediction.matrix[,l]))
      }
      
      vote.final <- levels(TY)[max.col(vote.matrix, ties.method="random")]
      
      
      prediction.correct[which(prediction.matrix[,l] == TY)] <- prediction.correct[which(prediction.matrix[,l] == TY)] + 1
      oob_err[l] <- mean(vote.final[which(oob.vector!=0)]!=TY[which(oob.vector!=0)])
      
      if(l > s){
        
        moob <- mean(oob_err[c((l-s):l)])  # Mittelwert des Fehlers der letzten s Bäume
        diffoob <- vector(mode = "numeric")  
        diffoob <- abs(oob_err[c((l-s):l)] - moob)  # absolute Abweichung der letzten s Bäume vom Mittelwert
        mdiffoob <- mean(diffoob)  # Mittelwert der Abweichung
        
        # convergence?
        if(abs(mdiffoob) < t){  # which value of t would be a good one?
          convergence <- TRUE
        }
      } 
      
      
      
      # weights: # trees where x is OOB and which predict correct value / # trees where x is OOB
      # for reliability concerns, only if already more than 20 trees in forest
      if((l > weight.threshold) && !convergence){
        
        weights[which(!is.na(prediction.matrix[,l]))] <- 1 - (prediction.correct[which(!is.na(prediction.matrix[,l]))]/oob.vector[which(!is.na(prediction.matrix[,l]))])
        
        
        # Normalisierung der Gewichte, sodass Summe = 1
        Z <- sum(weights)
        weights <- weights/Z
      }
      l <- l + 1
    }
    
    # Speichern der Ausgabeobjekte der Funktion
    result$num.trees <- l - 1
    result$oob.error <- oob_err
    result$type <- "classification"
    result$leaf.weights <- leaf.weights
    result$predictions <- prediction.matrix
    result$forest <- forest
    result$num.levels <- length(levels(TY))
    result$levels <- levels(TY)
    class(result) <- "brf"
    
    if(leaf.weights){
      result$weights <- weight.matrix
      result$majClassNodesForest <- majClassNodesForest
    }
    
    return(result) 
  }
}

predict.brf <- function(object, data, prob = FALSE){ 
  if(class(object)!="brf"){
    stop("object needs to be of class brf")
  }
  
  predictions.matrix <- vector(mode="numeric")
  
  if (object$type == "classification") {
    for(t in 1:(object$num.trees)){
      if(!object$leaf.weights){
        predictions.matrix <- cbind(predictions.matrix, predict(object$forest[[t]], data)$class)
      }
      if(object$leaf.weights){
        predNode <- predict(object$forest[[t]], data, membership = T)$membership # in welchem Endknoten landet Beobachtung i
        predClass <- apply(predNode,2, function(x) as.integer(object$majClassNodesForest[[t]][x]))  # welcher Klasse entspricht dieser Knoten in diesem Baum
        predictions.matrix <- cbind(predictions.matrix, predClass)
      }
    }
    if(!prob){
      predictions <- object$lev.names[as.numeric(apply(predictions.matrix, 1, function(x) names(which.max(table(x)))))]
      predictions <- factor(predictions, levels = object$lev.names)
      
      return(predictions)
    }
    
    if(prob){
      predictions.prob <- vector("numeric")
      for(i in 1:object$num.levels){
        predictions.prob <- cbind(predictions.prob, apply(predictions.matrix,1, function(x) length(which(x==i))))
      }
      predictions.prob <- predictions.prob/(object$num.trees)
      colnames(predictions.prob) <- object$lev.names
      return(as.matrix(predictions.prob))
    }
  }
}

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
