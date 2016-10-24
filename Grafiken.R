
install.packages("devtools")

devtools::install_github("openml/r")

install.packages("farff")
library(farff)

options(java.parameters = "- Xmx1024m") # Should avoid java gc overhead
library(OpenML)
saveOMLConfig(apikey = "28a8b4d5e774a8ccd18e78e6c6c00085", arff.reader = "farff", overwrite=TRUE) # den key kannst du nach der Accounterstellung auf der Website nachschauen; falls "farff" nicht geht mache "RWeka"
tasks = listOMLTasks()
tasktypes = listOMLTaskTypes()
clas = subset(tasks, task.type == "Supervised Classification") # über 1500 Klassifikationstasks

# Subset von 21 Tasks/Datasets
clas1 <- subset(clas, clas$task.id < 231 & 
                  clas$target.feature == "class" & clas$number.of.instances.with.missing.values == 0)
clas1 <- clas1[order(clas1$number.of.instances),]

clas1 <- clas1[which(clas1$task.id != 1),]

clas2 <- subset(clas1, clas1$number.of.instances %in% c(150,270,336,625,768,2000,1000000))
clas2 <- clas2[c(1:5,7:12,14:16,18,27),]
clas2 <- rbind(clas2, clas1[which(clas1$number.of.instances %in% c(10992, 45312, 295245, 829201)),])
clas2 <- clas2[order(clas2$number.of.instances),]



library(randomForestSRC)
task <- getOMLDataSet(data.id = 1, verbosity = 0)

#Task ID festhalten
dataID <- clas2$data.id
resultsRF <- matrix(NA, 20, 1500)
resultsBRF <- matrix(NA, 20, 1500)
resultsSTOP <- vector("numeric")
for(i in 1:20){
  runRF = matrix(NA, 50, 1500)
  runBRF = matrix(NA, 50, 1500)
  stop <- vector("numeric", length = 50)
  for(j in 1:50){
    print(c(i,j))
    task <- getOMLDataSet(data.id = dataID[i], verbosity = 0)
    task <- task$data
    set.seed(j)
    BRF = brf.conv(task$class, task[,-which(colnames(task)=="class")], forest.size = 1500, converge = F, stoptreeOut = T)
    set.seed(j)
    runRF[j,] = rfsrc(class~., data = task, ntree = 1500, tree.err = T)$err.rate[,1]
    runBRF[j,] = BRF$oob.error
    stop[j] = BRF$stoptree
  }
  runsRF = apply(runRF[c(1:10),], 2, mean)
  runsBRF = apply(runBRF[c(1:10),], 2, mean)
  stop = mean(stop, na.rm = T)
  resultsRF[i,] <- runsRF
  resultsBRF[i,] <- runsBRF
  resultsSTOP[i] <- stop
  plot(runsRF, ylim=c(min(runsBRF,runsRF),max(runsBRF,runsRF)), type = "l", col = "black", lwd = 1, xlab = "Anzahl Bäume", ylab = "OOB Fehlerrate", main = i)
  lines(runsBRF, col = "blue", lwd = 1)
  legend("topright", legend = c("RF", "BRF"), col = c("black", "blue"), lty = 1)
  abline(v=stoptree, col = "red")
  legend("topright", legend = c("RF", "BRF"), col = c("black", "blue"), lty = 1)
}
