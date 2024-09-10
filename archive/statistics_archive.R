# imports
library(lme4)
library(car)
library(emmeans)
library(glue)

# three-factor ANOVA
run_anova <- function(data_bsa, dv) {
  # treat categorical variables as factors
  data_bsa$subject <- factor(data_bsa$subject)
  data_bsa$strategy <- factor(data_bsa$strategy)
  data_bsa$l_length <- factor(data_bsa$l_length)
  data_bsa$pres_rate <- factor(data_bsa$pres_rate)
  
  formula_str <- paste(dv, '~ strategy*l_length + strategy*pres_rate')
  formula <- as.formula(formula_str)
  model <- lm(formula, data=data_bsa)
  res_anova <- Anova(model, type='III')
  
  # calculate mean squared errors
  res_anova$mse <- res_anova$`Sum Sq` / res_anova$Df
  
  return (list(res_anova=res_anova, model=model))
}

# post-hoc pairwise tests
post_hoc_pairwise <- function(res_anova, model) {
  me_pval <- res_anova["strategy", "Pr(>F)"]
  if (me_pval < 0.05) {
    emm <- emmeans(model, ~ strategy)
    res_tukey <- pairs(emm, adjust='tukey')
    return (res_tukey)
  } else {
    print('No significant main effect of strategy')
    return (NULL)
  }
}


# probability of 2nd recall intrusion following 1st recall intrusion
p2r_intr_data_bsa <- read.csv('analyses/dataframes/p2r_intr_data_bsa.csv')
p2r_intr_res <- run_anova(p2r_intr_data_bsa, 'intrusion')
p2r_intr_anova <- p2r_intr_res$res_anova
p2r_intr_model <- p2r_intr_res$model
write.csv(p2r_intr_anova, 'statistics/dataframes/p2r_intr_anova.csv')

p2r_intr_tukey_s <- post_hoc_pairwise(p2r_intr_anova, p2r_intr_model)
if (!is.null(p2r_intr_tukey_s)) {
  write.csv(as.data.frame(summary(p2r_intr_tukey_s)), 'statistics/dataframes/p2r_intr_tukey_s.csv',
            row.names=FALSE)
}

# lag-CRP (+1, -1 lag)
lcrp_data_bsa <- read.csv('analyses/dataframes/lcrp_data_bsa.csv')

lcrp_p1_res <- run_anova(lcrp_data_bsa, 'lp_1')
lcrp_p1_anova <- lcrp_p1_res$res_anova
lcrp_p1_model <- lcrp_p1_res$model
write.csv(lcrp_p1_anova, 'statistics/dataframes/lcrp_p1_anova.csv')

lcrp_p1_tukey_s <- post_hoc_pairwise(lcrp_p1_anova, lcrp_p1_model)
if (!is.null(lcrp_p1_tukey_s)) {
  write.csv(as.data.frame(summary(lcrp_p1_tukey_s)), 'statistics/dataframes/lcrp_p1_tukey_s.csv',
            row.names=FALSE)
}

lcrp_n1_res <- run_anova(lcrp_data_bsa, 'ln_1')
lcrp_n1_anova <- lcrp_n1_res$res_anova
lcrp_n1_model <- lcrp_n1_res$model
write.csv(lcrp_n1_anova, 'statistics/dataframes/lcrp_n1_anova.csv')

lcrp_n1_tukey_s <- post_hoc_pairwise(lcrp_n1_anova, lcrp_n1_model)
if (!is.null(lcrp_n1_tukey_s)) {
  write.csv(as.data.frame(summary(lcrp_n1_tukey_s)), 'statistics/dataframes/lcrp_n1_tukey_s.csv',
            row.names=FALSE)
}

# asymmetry (+1 v. -1, both available)
lcrp_l1_data_bsa <- read.csv('analyses/dataframes/lcrp_l1_data_bsa.csv')
lcrp_l1_res <- run_anova(lcrp_l1_data_bsa, 'crp_delta')
lcrp_l1_anova <- lcrp_l1_res$res_anova
lcrp_l1_model <- lcrp_l1_res$model
write.csv(lcrp_l1_anova, 'statistics/dataframes/lcrp_l1_anova.csv')

lcrp_l1_tukey_s <- post_hoc_pairwise(lcrp_l1_anova, lcrp_l1_model)
if (!is.null(lcrp_l1_tukey_s)) {
  write.csv(as.data.frame(summary(lcrp_l1_tukey_s)), 'statistics/dataframes/lcrp_l1_tukey_s.csv',
            row.names=FALSE)
}