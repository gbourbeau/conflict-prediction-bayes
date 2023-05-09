##### Thesis code for reproduction #####
## Predicting Conflict Using Bayesian Modeling: A Comparative Study of Cyprus, Mali, and Ukraine
## Gracen Bourbeau
## Columbia University, QMSS
## 10 May 2023

#note: data files avail. for download on github

##### MALI #####

##### Package Loading #####
library("tidyr"); library("dplyr")
library("magrittr"); library("ggplot2")
library("stringr"); library("readxl")
library("rstanarm"); library("rstan")
library("bayesplot"); library("loo");
library("arm"); library("tidyverse")

##### Data Import #####

mali <- read.csv("/2015-01-01-2023-03-01-Mali.csv",
                 header=T) #import of ACLED data

##### Data Cleaning #####

mali$event_date <- as.Date(mali$event_date, format="%d %B %Y") #formatting date variable

#did violence occur the next day?
mali$next_day_con <- ifelse(mali$event_date - lag(mali$event_date,n=1) ==-1,1,0)
# 1 = violence the next day

unique(mali$next_day_con)
table(mali$next_day_con)

#ridding of NA
mali<-na.omit(mali)

#coding response as factor
mali$next_day_con_fac <- as.factor(mali$next_day_con)
levels(mali$next_day_con_fac)[2]


##### additional data #####

#importing add'l political and social variables
mali_new_var <- read.csv("/mali_new_data - Sheet2.csv",
                          header=T)

# merging new variables with existing conflict data from ACLED
mali_merged_data <- merge(mali,mali_new_var,by="year")

#random sampling from entire population
n <- 100
data <- data.frame(x=runif(n), y=rnorm(n))
ind <- sample(c(TRUE, FALSE), n, replace=TRUE, prob=c(0.3, 0.7))
mali_random_samp <- mali_merged_data[ind, ]

sum(mali_random_samp$next_day_con==1)/length(mali_random_samp$next_day_con) #true val of conflict

##### rstanarm() model & loo #####

#full bayesian logistic reg model
mali_model_next_day_con <- stan_glm(next_day_con == levels(mali_random_samp$next_day_con_fac)[2] ~ latitude + longitude + fatalities + inflation + unemployment + military_gov_exp  + corruption + effectiveness + stability_violence + regulatory_quality + rule_of_law, 
                                        data = mali_random_samp,
                                        family=binomial(link="logit"),
                                        prior_intercept = normal(0,1),
                                        prior = laplace(location = 0, scale = 2.5, autoscale = FALSE))

summary(mali_model_next_day_con) #model summary
prior_summary(mali_model_next_day_con) #summary of priors

(loo_mali <- loo(mali_model_next_day_con,save_psis=TRUE)) #loo object

#save txt file with model summary
mali_model_next_day_con_summary <- summary(mali_model_next_day_con)
#write.table(mali_model_next_day_con_summ, file = "mali_all_var.txt", sep = ",", quote = FALSE, row.names = T)

##### plots ##### 
plot(mali_model_next_day_con,plotfun="areas")
pp_check(mali_model_next_day_con,plotfun="error_binned")+xlim(0,1)
plot(loo_mali,label_points=TRUE)
pp_check(mali_model_next_day_con,plotfun="error_binned",nreps=4)+xlim(0,1)
pp_check(mali_model_next_day_con, plotfun="ppc_dens_overlay")
pp_check(mali_model_next_day_con, plotfun = "stat_2d", stat = c("mean", "sd"))
pp_check(mali_model_next_day_con, plotfun="ppc_ecdf_overlay")

pplot<-plot(mali_model_next_day_con, "areas", prob = 0.95, prob_outer = 1)
pplot+ geom_vline(xintercept = 0)

binnedplot(mali_model_next_day_con$fitted.values,mali_model_next_day_con$residuals)

##### Predicted probabilities ##### 
linpred <- posterior_linpred(mali_model_next_day_con)
preds <- posterior_epred(mali_model_next_day_con)
pred <- colMeans(preds)
pr <- as.integer(pred >0.2836625) # percent posterior pred *over* true mean
round(mean(xor(pr,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo=E_loo(preds, loo_mali$psis_object, type="mean", log_ratios = -log_lik(mali_model_next_day_con))$value
round(mean(xor(ploo>0.2875071,as.integer(y==0))),2) # LOO classification accuracy *over* true mean

pr2 <- as.integer(pred<0.2836625) # percent posterior pred *below* true mean
round(mean(xor(pr2,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo2=E_loo(preds, loo_mali$psis_object, type="mean", log_ratios = -log_lik(mali_model_next_day_con))$value
round(mean(xor(ploo2<0.2875071,as.integer(y==0))),2) # LOO classification accuracy *below* true mean


##### prediction count ##### 
pcount <- posterior_predict(mali_model_next_day_con,draws=1)
pred <- colMeans(preds)
sum(pcount==1) # assigned as having next-day-conflict
length(pcount) # length model
sum(pcount==1)/length(pcount) #total conflict pred out of total (model)
sum(mali_random_samp$next_day_con==1)/length(mali_random_samp$next_day_con) #total conflict out of total (true)

##### comparing of intercept-only and original reduced model #####

#bayesian logistic reg model, reduced
mali_model_next_day_red <- stan_glm(next_day_con == levels(mali$next_day_con_fac)[2] ~ latitude + longitude + fatalities, 
                                  data = mali,
                                  family=binomial(link="logit"),
                                  prior_intercept = normal(0,1),
                                  prior = laplace(location = 0, scale = 0.15, autoscale = FALSE))

#bayesian logistic reg model, intercept only
mali_model_next_day_inter <- stan_glm(next_day_con == levels(mali$next_day_con_fac)[2] ~1, 
                                      data = mali,
                                      family=binomial(link="logit"),
                                      prior_intercept = NULL)

#loo generation for both models
(loo_mali_reduced <- loo(mali_model_next_day_red))
(loo_mali_intercept <- loo(mali_model_next_day_inter,save_psis=TRUE))

#comparing reduced model & intercept only model for ELPD to determine baseline
loo_compare(loo_mali_reduced,loo_mali_intercept)
loo_model_weights(list(reduced_mod=loo_mali_reduced, intercept_only=loo_mali_intercept))


##### CYPRUS #####


##### Package Loading #####
library("tidyr"); library("dplyr")
library("magrittr"); library("ggplot2")
library("stringr"); library("readxl")
library("rstanarm"); library("rstan")
library("bayesplot"); library("loo");
library("arm"); library("tidyverse")

##### Data Import #####

cyprus <- read.csv("/2015-01-01-2023-03-01-Cyprus.csv",
                   header=T) #import of ACLED data

##### Data Cleaning #####

cyprus$event_date <- as.Date(cyprus$event_date, format="%d %B %Y")

#did violence occur the next day?
cyprus$next_day_con <- ifelse(cyprus$event_date - lag(cyprus$event_date,n=1) ==-1,1,0)
# 1 = violence the next day

#ridding of NA
cyprus = cyprus[-1,]

#coding response as factor
cyprus$next_day_con_fac <- as.factor(cyprus$next_day_con)
levels(cyprus$next_day_con_fac)[2]

##### additional variables #####

#importing add'l political and social variables
cyprus_new_var <- read.csv("/cyprus_new_data - Sheet2 (1).csv",
                            header=T)

# merging new variables with existing conflict data from ACLED
cyprus_merged_data <- merge(cyprus,cyprus_new_var,by="year")

##### rstanarm() model and loo ##### 
#full bayesian logistic reg model
cyprus_model_next_day_con <- stan_glm(next_day_con == levels(cyprus_merged_data$next_day_con_fac)[2] ~ latitude + longitude + fatalities + inflation + unemployment + military_gov_exp  + corruption + effectiveness + stability_violence + regulatory_quality + rule_of_law + gini, 
                                          data = cyprus_merged_data,
                                          family=binomial(link="logit"),
                                          prior_intercept = normal(0,1),
                                          prior = laplace(location = 0, scale = 2.5, autoscale = FALSE))

summary(cyprus_model_next_day_con) #model summary
prior_summary(cyprus_model_next_day_con) #prior summary

(loo_cyprus <- loo(cyprus_model_next_day_con,save_psis=TRUE)) #loo object

##### plots ##### 
plot(cyprus_model_next_day_con,plotfun="areas")
pp_check(cyprus_model_next_day_con,plotfun="error_binned")+xlim(0,1)
plot(loo_cyprus,label_points=TRUE)
pp_check(cyprus_model_next_day_con,plotfun="error_binned",nreps=4)+xlim(0,1)
pp_check(cyprus_model_next_day_con, plotfun="ppc_dens_overlay")
pp_check(cyprus_model_next_day_con, plotfun = "stat_2d", stat = c("mean", "sd"))
pp_check(cyprus_model_next_day_con, plotfun="ppc_ecdf_overlay")

pplot<-plot(cyprus_model_next_day_con, "areas", prob = 0.95, prob_outer = 1)
pplot+ geom_vline(xintercept = 0)

binnedplot(cyprus_model_next_day_con$fitted.values,cyprus_model_next_day_con$residuals)

##### Predicted probabilities ##### 
linpred <- posterior_linpred(cyprus_model_next_day_con)
preds <- posterior_epred(cyprus_model_next_day_con)
pred <- colMeans(preds)
pr <- as.integer(pred >0.3239289) # percent posterior pred *over* true mean
round(mean(xor(pr,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo=E_loo(preds, loo_cyprus$psis_object, type="mean", log_ratios = -log_lik(cyprus_model_next_day_con))$value
round(mean(xor(ploo>0.3239289,as.integer(y==0))),2) # LOO classification accuracy *over* true mean

pr2 <- as.integer(pred<0.3239289) # percent posterior pred *below* true mean
round(mean(xor(pr2,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo2=E_loo(preds, loo_cyprus$psis_object, type="mean", log_ratios = -log_lik(cyprus_model_next_day_con))$value
round(mean(xor(ploo2<0.3239289,as.integer(y==0))),2) # LOO classification accuracy *below* true mean

##### prediction count ##### 
pcount <- posterior_predict(cyprus_model_next_day_con,draws=1)
pred <- colMeans(preds)
sum(pcount==1) # assigned as having next-day-conflict
length(pcount) # length model
sum(pcount==1)/length(pcount) #total conflict pred out of total (model)
sum(cyprus_merged_data$next_day_con==1)/length(cyprus_merged_data$next_day_con) #total conflict out of total (true)

##### comparing of intercept-only and original reduced model #####

#bayesian logistic reg model, reduced
cyprus_model_next_day_red <- stan_glm(next_day_con == levels(cyprus$next_day_con_fac)[2] ~ latitude + longitude + fatalities, 
                                    data = cyprus,
                                    family=binomial(link="logit"),
                                    prior_intercept = normal(0,1),
                                    prior = laplace(location = 0, scale = 2.5, autoscale = FALSE))

#bayesian logistic reg model, intercept only
cyprus_model_next_day_inter <- stan_glm(next_day_con == levels(cyprus$next_day_con_fac)[2] ~1, 
                                        data = cyprus,
                                        family=binomial(link="logit"),
                                        prior_intercept = NULL)

#loo generation for both models
(loo_cyprus_reduced <- loo(cyprus_model_next_day_red))
(loo_cyprus_intercept <- loo(cyprus_model_next_day_inter,save_psis=TRUE))

#comparing reduced model & intercept only model for ELPD to determine baseline
loo_compare(loo_cyprus_reduced,loo_cyprus_intercept)
loo_model_weights(list(reduced_mod=loo_cyprus_reduced, intercept_only=loo_cyprus_intercept))


##### UKRAINE #####


##### Package Loading #####
library("tidyr"); library("dplyr")
library("magrittr"); library("ggplot2")
library("stringr"); library("readxl")
library("rstanarm"); library("rstan")
library("bayesplot"); library("loo");
library("arm"); library("tidyverse")

##### Data Import #####


ukraine <- read.csv("/2015-01-01-2023-03-01-Ukraine.csv",
                    header=T) #import of ACLED data

##### Data Cleaning #####
ukraine$event_date <- as.Date(ukraine$event_date, format="%d %B %Y")

#did violence occur the next day?
ukraine$next_day_con <- ifelse(ukraine$event_date - lag(ukraine$event_date,n=1) ==-1,1,0)
# 1 = violence the next day

#ridding of NA
ukraine<-na.omit(ukraine)

#coding response as factor
ukraine$next_day_con_fac <- as.factor(ukraine$next_day_con)

##### additional variables ##### 

#importing add'l political and social variables
ukr_new_var <- read.csv("/ukraine_new_data - Sheet2.csv",
                         header=T)

# merging new variables with existing conflict data from ACLED
ukr_merged_data <- merge(ukraine,ukr_new_var,by="year")

#random sampling from entire population
n <- 100
data <- data.frame(x=runif(n), y=rnorm(n))
ind <- sample(c(TRUE, FALSE), n, replace=TRUE, prob=c(0.1, 0.9))
ukr_random_sample <- ukr_merged_data[ind, ]

sum(ukr_random_sample$next_day_con==1)/length(ukr_random_sample$next_day_con) #true val conflict

##### rstanarm() model & loo ##### 
#full bayesian logistic reg model
ukr_model_next_day_con <- stan_glm(next_day_con == levels(ukr_random_sample$next_day_con_fac)[2] ~ latitude + longitude + fatalities + inflation + unemployment + military_gov_exp  + corruption + effectiveness + stability_violence + regulatory_quality + rule_of_law, 
                                        data = ukr_random_sample,
                                        family=binomial(link="logit"),
                                        prior_intercept = normal(0,1),
                                        adapt_delta=0.99,
                                        prior = laplace(location = 0, scale = 2.5, autoscale = FALSE))

summary(ukr_model_next_day_con) #model summary
prior_summary(ukr_model_next_day_con) #prior summary

(loo_ukraine <- loo(ukr_model_next_day_con,save_psis=TRUE,k_threshold = 0.7)) #loo object

#save txt file with model summary
ukr_model_next_day_all_var_summ <- summary(ukr_model_next_day_con)
#write.table(ukr_model_next_day_all_var_summ, file = "ukr_all_var.txt", sep = ",", quote = FALSE, row.names = T)

##### plots ##### 
plot(ukr_model_next_day_con,plotfun="areas")
pp_check(ukr_model_next_day_con,plotfun="error_binned")+xlim(0,1)
plot(loo_ukraine,label_points=TRUE)
pp_check(ukr_model_next_day_con,plotfun="error_binned",nreps=4)+xlim(0,1)
pp_check(ukr_model_next_day_con, plotfun="ppc_dens_overlay")
pp_check(ukr_model_next_day_con, plotfun = "stat_2d", stat = c("mean", "sd"))
pp_check(ukr_model_next_day_con, plotfun="ppc_ecdf_overlay")

pplot<-plot(ukr_model_next_day_con, "areas", prob = 0.95, prob_outer = 1)
pplot+ geom_vline(xintercept = 0)

binnedplot(ukr_model_next_day_con$fitted.values,ukr_model_next_day_con$residuals)

##### Predicted probabilities #####
##### Predicted probabilities ##### 
linpred <- posterior_linpred(ukr_model_next_day_con)
preds <- posterior_epred(ukr_model_next_day_con)
pred <- colMeans(preds)
pr <- as.integer(pred >0.02584493) # percent posterior pred *over* true mean
round(mean(xor(pr,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo=E_loo(preds, loo_ukraine$psis_object, type="mean", log_ratios = -log_lik(ukr_model_next_day_con))$value
round(mean(xor(ploo>0.02584493,as.integer(y==0))),2) # LOO classification accuracy *over* true mean

pr2 <- as.integer(pred<0.02584493) # percent posterior pred *below* true mean
round(mean(xor(pr2,as.integer(y==0))),2) # posterior classification accuracy

# LOO predictive probabilities
ploo2=E_loo(preds, loo_ukraine$psis_object, type="mean", log_ratios = -log_lik(ukr_model_next_day_con))$value
round(mean(xor(ploo2<0.02584493,as.integer(y==0))),2) # LOO classification accuracy *below* true mean

##### prediction count ##### 
pcount <- posterior_predict(ukr_model_next_day_con,draws=1)
pred <- colMeans(preds)
sum(pcount==1) # assigned as having next-day-conflict
length(pcount) # length model
sum(pcount==1)/length(pcount) #total conflict pred out of total (model)
sum(ukr_random_sample$next_day_con==1)/length(ukr_random_sample$next_day_con) #total conflict out of total (true)

##### comparing of intercept-only and original reduced model #####

#subsetting, random sampling dataset
n <- 100
data <- data.frame(x=runif(n), y=rnorm(n))
ind <- sample(c(TRUE, FALSE), n, replace=TRUE, prob=c(0.1, 0.9))
ukr_subset <- ukraine[ind, ]

#bayesian logistic reg model, reduced 
ukraine_model_next_day_red <- stan_glm(next_day_con == levels(ukr_subset$next_day_con_fac)[2] ~ latitude + longitude + fatalities, 
                                     data = ukr_subset,
                                     family=binomial(link="logit"),
                                     prior_intercept = normal(0,1),
                                     adapt_delta=0.99,
                                     prior = laplace(location = 0, scale = 0.15, autoscale = FALSE))

#bayesian logistic reg model, intercept only
ukraine_model_next_inter <- stan_glm(next_day_con == levels(ukr_subset$next_day_con_fac)[2] ~ 1, 
                                     data = ukr_subset,
                                     family=binomial(link="logit"),
                                     adapt_delta=0.99,
                                     prior_intercept = NULL)

#loo generation for both models
(loo_ukr_reduced <- loo(ukraine_model_next_day_red))
(loo_ukr_intercept <- loo(ukraine_model_next_inter,save_psis=TRUE))

#comparing reduced model & intercept only model for ELPD to determine baseline
loo_compare(loo_ukr_reduced,loo_ukr_intercept)
loo_model_weights(list(reduced_mod=loo_ukr_reduced, intercept_only=loo_ukr_intercept))
