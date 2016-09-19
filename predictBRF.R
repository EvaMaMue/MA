#library(data.table)

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

pred.test <- brf.predict(test, iris, prob = T)

