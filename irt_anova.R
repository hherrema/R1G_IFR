# imports
library(lme4)
library(car)

# 4-factor ANOVA
# inter-response times for final 4 transitions
run_irt_anova <- function(irt_data) {
  # treat categorical variables as factors
  irt_data$subject <- factor(irt_data$subject)
  irt_data$strategy <- factor(irt_data$strategy)
  irt_data$rot <- factor(irt_data$rot)
  irt_data$l_length <- factor(irt_data$l_length)
  irt_data$pres_rate <- factor(irt_data$pres_rate)
  
  model <- lm(irt ~ strategy + rot + l_length + pres_rate + strategy:rot + strategy:l_length + strategy:pres_rate + rot:l_length + rot:pres_rate, data=irt_data)
  res_anova <- Anova(model, type='III')
  
  # calculate mean squared errors
  res_anova$mse <- res_anova$`Sum Sq` / res_anova$Df
  
  return (res_anova)
}

irt_data <- read.csv('analyses/dataframes/irt_final_4_data_bsa.csv')
irt_anova <- run_irt_anova(irt_data)
write.csv(irt_anova, 'statistics/dataframes/irt_anova.csv')
