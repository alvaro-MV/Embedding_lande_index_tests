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
  conversation = as.double(strsplit(str, split = " ")[[1]])
  lista[[count]] = conversation
  count  = count + 1
}

## Meter una linea
plot_conversation <- function(conversation){
  plot(conversation, col="darkgrey")
  lines(lowess(time(conversation), conversation), col="blue", lwd=2)
}

# acf(lista[[2]])

plot_acf_pacf_list <- function(series_list,
                               names      = NULL,
                               max_lag    = NULL,
                               cols       = 2,
                               col_acf    = "#1f77b4",
                               col_pacf   = "#d62728",
                               na_rm      = TRUE,     # ← nuevo
                               file       = NULL) {   # ← opcional: "fig.pdf" o "fig.png"
  # series_list : lista de vectores numéricos
  # names       : etiquetas de las series (opcional)
  # max_lag     : máximo retardo
  # cols        : 2 ⇒ ACF y PACF lado a lado
  # file        : si se especifica, guarda la figura en PDF/PNG

  n_series <- length(series_list)
  if (is.null(names)) names <- paste("Series", seq_len(n_series))
  if (is.null(max_lag))
      max_lag <- floor(max(vapply(series_list, length, 1L)) / 4)

  # -------- abrir dispositivo gráfico ----------
  if (!is.null(file)) {
    ext <- tools::file_ext(file)
    if      (ext == "pdf") pdf(file,  width = 4*cols, height = 2.5*n_series)
    else if (ext %in% c("png", "jpeg", "jpg", "tiff")) {
      do.call(ext, list(file = file, width = 1000*cols, height = 600*n_series,
                        res = 150))
    } else stop("Extensión de archivo no reconocida: usa pdf, png, jpg, tiff…")
  }

  op <- par(no.readonly = TRUE)
  on.exit({
    par(op)
    if (!is.null(file)) dev.off()
  })

  par(mfrow = c(n_series, cols), mar = c(3.5, 3.5, 2, 1), oma = c(0, 0, 2, 0))

  for (i in seq_along(series_list)) {
    s <- as.numeric(series_list[[i]])
    if (na_rm) s <- s[!is.na(s)]
    if (length(s) < 2) {
      warning(sprintf("Serie %s demasiado corta después de eliminar NA.", names[i]))
      next
    }

    acf( s, lag.max = max_lag, na.action = na.pass,
         main = paste(names[i], "– ACF"),  col = col_acf, lwd = 2)

    pacf(s, lag.max = max_lag, na.action = na.pass,
         main = paste(names[i], "– PACF"), col = col_pacf, lwd = 2, ylab = "")
  }

  mtext("Autocorrelation structure of Lande-index series",
        outer = TRUE, cex = 1.2, line = 0.5)
}

# ─── Ejemplos ───────────────────────────────────────────────────────
# Pantalla (elimina NA por defecto):
plot_acf_pacf_list(lista, names = names(lista), max_lag = 12)

# Guardar en PDF:
plot_acf_pacf_list(lista, names = names(lista),
                   max_lag = 12, file = "acf_pacf_panel.pdf")


## ── Ejemplo de uso ───────────────────────────────────────────────
##  lista es una list donde cada elemento es un vector de dobles
##  names(lista) opcionalmente contiene los nombres de cada serie

plot_acf_pacf_list(
  series_list = lista,
  names       = names(lista),    # o NULL
  max_lag     = 12               # o deja que lo calcule
)

for (i in seq_along(lista)) {
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
