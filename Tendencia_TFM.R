
install.packages("Kendall")
install.packages("rjson")
require(Kendall)
require(rjson)
library(stringr)

## Tratamiento de datos
myData <- read.csv('tendencias_global', header = TRUE)
View(myData)

list <- vector(mode = "list", length = nrow(myData))
count = 1
for (i in myData['lande_conversation'][[1]]){
  str = substr(i, start = 2, stop = nchar(i)-1) 
  conversation = as.double(strsplit(str, split = ", ")[[1]])
  list[[count]] = conversation
  count  = count + 1
  }


## Meter una linea
plot_conversation <- function(conversation){
  plot(conversation, col="darkgrey")
  lines(lowess(time(conversation), conversation), col="blue", lwd=2)
}

conversation = list[[9]]
acf(conversation)

plot_autocorrelation <- function(conversation){
  ## Graficos de autocorrelacion
  
  par(mfrow=c(2,1))
  acf(conversation)
  pacf(conversation)
}
plot_autocorrelation(list[[1]])

### MannKendall
for (i in 1:length(list)){
  print(MannKendall(list[[i]]))
}
summary(MannKendall(list[[3]]))
