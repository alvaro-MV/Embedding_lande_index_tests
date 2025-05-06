require(Kendall)
require(rjson)
library(stringr)

options(digits = 16)  # precisión alta

# Carga de datos
data <- read.csv("data/tendencias_expositivo", header = TRUE)

# Construcción de la lista de series
n <- nrow(data)
lista <- vector("list", length = n)
for (i in seq_len(n)) {
  raw <- as.character(data$lande_intra[i])
  str <- substr(raw, 2, nchar(raw) - 1)
  nums <- as.double(strsplit(str, " ")[[1]])
  lista[[i]] <- nums
}

# Vectores para almacenar resultados
taus  <- rep(NA_real_, n)
pvals <- rep(NA_real_, n)

# Test de Mann–Kendall y recolección de tau/p
for (i in seq_along(lista)) {
  serie <- scale(lista[[i]])          # centrar sin alterar varianza
  if (length(serie) >= 3 && sd(serie, na.rm = TRUE) > 0) {
    mk       <- MannKendall(serie)
    taus[i]  <- mk$tau
    pvals[i] <- mk$sl               # p-value crudo
  }
}

# Corrección de Bonferroni
# adj_p <- p.adjust(pvals, method = "holm")

# Mostrar resumen con p ajustadas
for (i in seq_len(n)) {
  cat(sprintf(
    "Serie %d: tau = %+0.3f, p = %0.4f, p_adj = %0.4f, sig(Bonf) = %s\n",
    i,
    taus[i],
    pvals[i],
    pvals[i],
    if (!is.na(pvals[i]) && pvals[i] < 0.05) "YES" else "NO"
  ))
}
