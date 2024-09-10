# imports
library(lme4)
library(car)

# recall initiation bias
r1_data <- read.csv('analyses/dataframes/prim_rec_pfr.csv')

# treat categorical variables as factors
r1_data$subject <- factor(r1_data$subject)
r1_data$l_length <- factor(r1_data$l_length)
r1_data$pres_rate <- factor(r1_data$pres_rate)

# 2-factor ANOVA
model <- lm(rec_prim_bias ~ l_length + pres_rate, data=r1_data)
r1_anova <- Anova(model, type='III')

# calculate mean squared errors
r1_anova$mse <- r1_anova$`Sum Sq` / r1_anova$Df

# save out results
write.csv(r1_anova, 'statistics/dataframes/r1_anova.csv')
