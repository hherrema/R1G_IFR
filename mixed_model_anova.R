# imports
library(rstatix)
library(glue)

# mixed-model ANOVA
run_ns_anova <- function(ns_data_bsa, dv) {
  # treat categorical variables as factors
  ns_data_bsa$subject <- factor(ns_data_bsa$subject)
  ns_data_bsa$r1_label <- factor(ns_data_bsa$r1_label)
  ns_data_bsa$l_length <- factor(ns_data_bsa$l_length)
  ns_data_bsa$pres_rate <- factor(ns_data_bsa$pres_rate)
  
  formula_str <- paste(dv, '~ r1_label*l_length + r1_label*pres_rate')
  formula <- as.formula(formula_str)
  model <- anova_test(data=ns_data_bsa, formula=formula, wid=subject,
                      between=c(l_length, pres_rate), within=(r1_label), type=3)
  res_anova <- get_anova_table(model)
  
  return (res_anova)
}


# mean words recalled
mwr_ns_data_bsa <- read.csv('analyses/dataframes/mwr_ns_data_bsa.csv')
mwr_ns_anova <- run_ns_anova(mwr_ns_data_bsa, 'mwr')
write.csv(mwr_ns_anova, 'statistics/dataframes/mwr_ns_anova.csv')


# initial response times
rti_ns_data_bsa <- read.csv('analyses/dataframes/rti_ns_data_bsa.csv')
rti_ns_anova <- run_ns_anova(rti_ns_data_bsa, 'rt')
write.csv(rti_ns_anova, 'statistics/dataframes/rti_ns_anova.csv')


# temporal clustering scores
tcl_ns_data_bsa <- read.csv('analyses/dataframes//tcl_ns_data_bsa.csv')
tcl_ns_anova <- run_ns_anova(tcl_ns_data_bsa, 'tcl')
write.csv(tcl_ns_anova, 'statistics/dataframes/tcl_ns_anova.csv')
