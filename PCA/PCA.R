df_y_x <- read.csv('train.csv')
x <- df_y_x[-1]

#means by column
col_means <- vector()
for(name in colnames(x)){
  col_means <- c(col_means, mean(x[[name]]) )
}

#subtract mean from everu column
l <- length(x)
for(i in 1:l){
  x[[i]] <- x[[i]]-col_means[i]
}

#variance for every column
col_var <- vector()
for(name in colnames(x)){
  col_var <- c(col_var, sqrt(sum(x[[name]]^2)))
}

#normalization of every column
for(i in 1:l){
  x[[i]] <- x[[i]]/col_var[i]
}

corr_f <- cor(x)

corr_matr <- t(as.matrix(x))%*%as.matrix(x)


