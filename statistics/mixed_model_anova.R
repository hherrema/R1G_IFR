# imports
library(rstatix)
library(glue)

# mixed-model ANOVA
run_ns_anova <- function(ns_data_bsa_long, dv) {
  # treat categorical variables as factors
  ns_data_bsa_long$subject <- factor(ns_data_bsa_long$subject)
  ns_data_bsa_long$r1_label <- factor(ns_data_bsa_long$r1_label)
  ns_data_bsa_long$l_length <- factor(ns_data_bsa_long$l_length)
  ns_data_bsa_long$pres_rate <- factor(ns_data_bsa_long$pres_rate)
  
  formula_str <- paste(dv, '~ r1_label*l_length + r1_label*pres_rate')
  formula <- as.formula(formula_str)
  model <- anova_test(data=ns_data_bsa_long, formula=formula, wid=subject,
                      between=c(l_length, pres_rate), within=(r1_label), type=3)
  res_anova <- get_anova_table(model)
  return (res_anova)
}

# mean words recalled
mwr_ns_data_bsa_long <- read.csv('dataframes/analyses/mwr_ns_data_bsa_long.csv')
mwr_ns_anova <- run_ns_anova(mwr_ns_data_bsa_long, 'mwr')
write.csv(mwr_ns_anova, 'statistics/mwr_ns_anova.csv')

# ELI rates
eli_ns_data_bsa_long <- read.csv('dataframes/analyses/eli_ns_data_bsa_long.csv')
eli_ns_anova <- run_ns_anova(eli_ns_data_bsa_long, 'eli')
write.csv(eli_ns_anova, 'statistics/eli_ns_anova.csv')

# initial response times
rti_ns_data_bsa_long <- read.csv('dataframes/analyses/rti_ns_data_bsa_long.csv')
rti_ns_anova <- run_ns_anova(rti_ns_data_bsa_long, 'rt')
write.csv(rti_ns_anova, 'statistics/rti_ns_anova.csv')

# temporal clustering scores
tcl_ns_data_bsa_long <- read.csv('dataframes/analyses/tcl_ns_data_bsa_long.csv')
tcl_ns_anova <- run_ns_anova(tcl_ns_data_bsa_long, 'tcl')
write.csv(tcl_ns_anova, 'statistics/tcl_ns_anova.csv')

# lag-CRP (+1, -1 lag)
lcrp_p1_ns_data_bsa_long <- read.csv('dataframes/analyses/lcrp_p1_ns_data_bsa_long.csv')
lcrp_p1_ns_anova <- run_ns_anova(lcrp_p1_ns_data_bsa_long, 'crp')
write.csv(lcrp_p1_ns_anova, 'statistics/lcrp_p1_ns_anova.csv')

lcrp_n1_ns_data_bsa_long <- read.csv('dataframes/analyses/lcrp_n1_ns_data_bsa_long.csv')
lcrp_n1_ns_anova <- run_ns_anova(lcrp_n1_ns_data_bsa_long, 'crp')
write.csv(lcrp_n1_ns_anova, 'statistics/lcrp_n1_ns_anova.csv')
