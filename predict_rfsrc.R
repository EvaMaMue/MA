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
