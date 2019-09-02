#* Load libraries
library(boot)

#* Create a random (normal distribution)
set.seed(10)
x <- rnorm(n = 100, mean = 0, sd = 1)

#* Determine confidence interval from t distribution (-0.3233, 0.0502)
t.test(
  x = x
)$conf.int

#* Derive confidence interval (assume normal distribution) (-0.3210, 0.0479)
mean(x) + sd(x)/sqrt(length(x)) * qnorm(p = 0.975)
mean(x) - sd(x)/sqrt(length(x)) * qnorm(p = 0.975)


#****************************
##### Bootstrap Results #####
#****************************

mean_fun <- function(x, i){
  
  return(mean( x[i] ))
  
}

#* Use boot library for bootstrapping the mean
set.seed(10)
Boot_Mean <- boot(
  data = x
  , statistic = mean_fun
  , R = 1000
  )

#* Determine the bootstrap confidence interval for the mean (-0.3213, 0.0435)
boot.ci(Boot_Mean)


#* Build my own bootstrap
n <- 1000
bootstrap_matrix <- NULL

set.seed(10)
for (i in 1:n){
  
  # Iteratively append resampled vectors
  bootstrap_matrix <- rbind(
    bootstrap_matrix
    , sample(
      x = x
      , size = length(x)
      , replace = TRUE
      )
  )
  
  i = i + 1

} # Close loop


#* Store the variance between sample mean and bootstrap means
bootstrap_var <- rowMeans(x = bootstrap_matrix) - mean(x)


#* Lower limit
mean(x) - quantile(
  x = bootstrap_var
  , probs = 0.975
)

#* Upper limit
mean(x) - quantile(
  x = bootstrap_var
  , probs = 0.025
)