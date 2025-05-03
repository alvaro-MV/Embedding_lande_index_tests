# install.packages("Kendall")
# install.packages("rjson")
require(Kendall)
require(rjson)
library(stringr)

options(digits = 16)  # or even higher if needed

## Tratamiento de datos
myData <- read.csv('data/tendencias_expositivo', header = TRUE)
View(myData)

lista <- vector(mode = "list", length = nrow(myData))
count = 1
for (i in myData['lande_intra'][[1]]){
  str = substr(i, start = 2, stop = nchar(i)-1) 
  conversation = as.double(strsplit(str, split = ", ")[[1]])
  lista[[count]] = conversation
  count  = count + 1
}

## Meter una linea
plot_conversation <- function(conversation){
  plot(conversation, col="darkgrey")
  lines(lowess(time(conversation), conversation), col="blue", lwd=2)
}

# acf(lista[[2]])

plot_autocorrelation <- function(lande){
  ## Graficos de autocorrelacion
  
  par(mfrow=c(2,1))
  acf(lande)
  pacf(lande)
}
# plot_autocorrelation(lista[[1]])

for (i in 1:length(lista)) {
  serie <- lista[[i]]
  serie <- scale(serie)
  
  # Evita errores por longitud o falta de variación
  if (length(serie) >= 3 && sd(serie, na.rm = TRUE) > 0) {
    print(paste("Serie", i))
    print(MannKendall(serie))
  } else {
    print(paste("Serie", i, "omitida por falta de datos o variación."))
  }
}
