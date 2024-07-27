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


# mean words recalled
mwr_data_bsa <- read.csv('dataframes/analyses/mwr_data_bsa.csv')
mwr_res <- run_anova(mwr_data_bsa, 'mwr')
mwr_anova <- mwr_res$res_anova
mwr_model <- mwr_res$model
write.csv(mwr_anova, 'statistics/mwr_anova.csv')

mwr_tukey_s <- post_hoc_pairwise(mwr_anova, mwr_model)
if (!is.null(mwr_tukey_s)) {
  write.csv(as.data.frame(summary(mwr_tukey_s)), 'statistics/mwr_tukey_s.csv', 
            row.names=FALSE)
}

# ELI and PLI rates
intr_data_only_cr_bsa <- read.csv('dataframes/analyses/intr_data_only_cr_bsa.csv')

eli_res <- run_anova(intr_data_only_cr_bsa, 'eli_rate')
eli_anova <- eli_res$res_anova
eli_model <- eli_res$model
write.csv(eli_anova, 'statistics/eli_anova.csv')

eli_tukey_s <- post_hoc_pairwise(eli_anova, eli_model)
if (!is.null(eli_tukey_s)) {
  write.csv(as.data.frame(summary(eli_tukey_s)), 'statistics/eli_tukey_s.csv',
            row.names=FALSE)
}

pli_res <- run_anova(intr_data_only_cr_bsa, 'pli_rate')
pli_anova <- pli_res$res_anova
pli_model <- pli_res$model
write.csv(pli_anova, 'statistics/pli_anova.csv')

pli_tukey_s <- post_hoc_pairwise(pli_anova, pli_model)
if (!is.nul(pli_tukey_s)) {
  write.csv(as.data.frame(summary(pli_tukey_s)), 'statistics/pli_tukey_s.csv',
            row.names=FALSE)
}

# R1 intrusions
r1_intr_data_bsa <- read.csv('dataframes/analyses/r1_intr_data_bsa.csv')
r1_intr_res <- run_anova(r1_intr_data_bsa, 'prop_wrong')
r1_intr_anova <- r1_intr_res$res_anova
r1_intr_model <- r1_intr_res$model
write.csv(r1_intr_anova, 'statistics/r1_intr_anova.csv')

r1_intr_tukey_s <- post_hoc_pairwise(r1_intr_anova, r1_intr_model)
if (!is.null(r1_intr_tukey_s)) {
  write.csv(as.data.frame(summary(r1_intr_tukey_s)), 'statistics/r1_intr_tukey_s.csv',
            row.names=FALSE)
}

# probability of 2nd recall intrusion following 1st recall intrusion
p2r_intr_data_bsa <- read.csv('dataframes/analyses/p2r_intr_data_bsa.csv')
p2r_intr_res <- run_anova(p2r_intr_data_bsa, 'intrusion')
p2r_intr_anova <- p2r_intr_res$res_anova
p2r_intr_model <- p2r_intr_res$model
write.csv(p2r_intr_anova, 'statistics/p2r_intr_anova.csv')

p2r_intr_tukey_s <- post_hoc_pairwise(p2r_intr_anova, p2r_intr_model)
if (!is.null(p2r_intr_tukey_s)) {
  write.csv(as.data.frame(summary(p2r_intr_tukey_s)), 'statistics/p2r_intr_tukey_s.csv',
            row.names=FALSE)
}

# initial response times
rti_data_only_cr_bsa <- read.csv('dataframes/analyses/rti_data_only_cr_bsa.csv')
rti_res <- run_anova(rti_data_only_cr_bsa, 'rt')
rti_anova <- rti_res$res_anova
rti_model <- rti_res$model
write.csv(rti_anova, 'statistics/rti_anova.csv')

rti_tukey_s <- post_hoc_pairwise(rti_anova, rti_model)
if (!is.null(rti_tukey_s)) {
  write.csv(as.data.frame(summary(rti_tukey_s)), 'statistics/rti_tukey_s.csv',
            row.names=FALSE)
}

# temporal clustering score
tcl_data_bsa <- read.csv('dataframes/analyses/tcl_data_bsa.csv')
tcl_res <- run_anova(tcl_data_bsa, 'tcl')
tcl_anova <- tcl_res$res_anova
tcl_model <- tcl_res$model
write.csv(tcl_anova, 'statistics/tcl_anova.csv')

tcl_tukey_s <- post_hoc_pairwise(tcl_anova, tcl_model)
if (!is.null(tcl_tukey_s)) {
  write.csv(as.data.frame(summary(tcl_tukey_s)), 'statistics/tcl_tukey_s.csv',
            row.names=FALSE)
}

# semantic clustering score
scl_data_bsa <- read.csv('dataframes/analyses/scl_data_bsa.csv')
scl_res <- run_anova(scl_data_bsa, 'scl')
scl_anova <- scl_res$res_anova
scl_model <- scl_res$model
write.csv(scl_anova, 'statistics/scl_anova.csv')

scl_tukey_s <- post_hoc_pairwise(scl_anova, scl_model)
if (!is.null(scl_tukey_s)) {
  write.csv(as.data.frame(summary(scl_tukey_s)), 'statistics/scl_tukey_s.csv',
            row.names=FALSE)
}

# lag-CRP (+1, -1 lag)
lcrp_data_bsa <- read.csv('dataframes/analyses/lcrp_data_bsa.csv')

lcrp_p1_res <- run_anova(lcrp_data_bsa, 'lp_1')
lcrp_p1_anova <- lcrp_p1_res$res_anova
lcrp_p1_model <- lcrp_p1_res$model
write.csv(lcrp_p1_anova, 'statistics/lcrp_p1_anova.csv')

lcrp_p1_tukey_s <- post_hoc_pairwise(lcrp_p1_anova, lcrp_p1_model)
if (!is.null(lcrp_p1_tukey_s)) {
  write.csv(as.data.frame(summary(lcrp_p1_tukey_s)), 'statistics/lcrp_p1_tukey_s.csv',
            row.names=FALSE)
}

lcrp_n1_res <- run_anova(lcrp_data_bsa, 'ln_1')
lcrp_n1_anova <- lcrp_n1_res$res_anova
lcrp_n1_model <- lcrp_n1_res$model
write.csv(lcrp_n1_model, 'statistics/lcrp_n1_anova.csv')

lcrp_n1_tukey_s <- post_hoc_pairwise(lcrp_n1_anova, lcrp_n1_model)
if (!is.null(lcrp_n1_tukey_s)) {
  write.csv(as.data.frame(summary(lcrp_n1_tukey_s)), 'statistics/lcrp_n1_tukey_s.csv',
            row.names=FALSE)
}

# 5-factor Anova (inter-response times)
run_irt_anova <- function() {
  irt_data_bsa <- read.csv('dataframes/analyses/irt_data_bsa.csv')
  
  irt_data_bsa$subject <- factor(irt_data_bsa$subject)
  irt_data_bsa$strategy <- factor(irt_data_bsa$strategy)
  irt_data_bsa$l_length <- factor(irt_data_bsa$l_length)
  irt_data_bsa$pres_rate <- factor(irt_data_bsa$pres_rate)
  irt_data_bsa$ntr <- factor(irt_data_bsa$ntr)
  irt_data_bsa$output_transition <- factor(irt_data_bsa$output_transition)
  
  model <- lm(irt ~ strategy*ntr*output_transition + strategy*l_length + strategy*pres_rate, data=irt_data_bsa)
  res_anova <- Anova(model, type='III')
  
  write.csv(res_anova, 'statistics/irt_anova.csv')
}
