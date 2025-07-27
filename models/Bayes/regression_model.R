library(brms)
library(car)
library(ggplot2)
library(dplyr)
library(posterior)
library(rstan)

# -----------------------------
# Read and pre-process the data
# -----------------------------

# Read the csv file
data <- read.csv("atoll_IME_MHW_summary.csv")

# Log-transformation
data$log_IME_chl <- log(data$IME_chl)
data$log_area <- log(data$total_atoll_area_sqkm)

# Create a categorical variable
data$channel_class <- cut(data$channel_width,
                          breaks = c(-Inf, 0, Inf),
                          labels = c("none", "open"))

# Explicitly set the reference level
data$channel_class <- relevel(data$channel_class, ref = "none")

# Check for multicollinearity
car::vif(lm(log_IME_chl ~ log_area + channel_class,
            data = data))

# Check data distribution
hist(data$log_IME_chl, breaks = 20, probability = TRUE, main = "Histogram")
curve(dnorm(x, mean = mean(data$log_IME_chl), sd = sd(data$log_IME_chl)), 
      col = "red", lwd = 2, add = TRUE)

# Check for NA
colSums(is.na(data))

# --------------
# Fit the model
# --------------

formula_IME <- bf(log_IME_chl ~ log_area + channel_class)

model_IME <- brm(formula = formula_IME,
                    data = data,
                    family = gaussian(),
                    chains = 4,
                    warmup = 5000,
                    iter = 10000,
                    cores = 4,
                    control = list(adapt_delta = 0.95, max_treedepth = 12),
                    save_pars = save_pars(all = TRUE))

# ------------------------------
# Model diagnostics and results
# ------------------------------

plot(model_IME)
pp_check(model_IME, type = "dens_overlay")
pp_check(model_IME, type = "scatter_avg")
pp_check(model_IME, type = "stat_2d")
pp_check(model_IME, type = "ribbon")
pp_check(model_IME, type = "stat", stat = "var")

loo(model_IME, momen_match = TRUE) # leave-one-out cross-validation

coda::gelman.diag(as.mcmc(model_IME)[, 1:7]) # ideally all close to one
coda::geweke.diag(as.mcmc(model_IME)[, 1:7]) # ideally all values between -2 and +2

summary(model_IME) # Check R_hat, ESS_bulk, ESS_tail

# Compute BFMI
stanfit <- model_IME$fit

# Extract sampler parameters
sampler_params <- get_sampler_params(stanfit, inc_warmup = FALSE)

# Compute BFMI for each chain
bfmi <- sapply(sampler_params, function(chain) {
  numer <- mean(diff(chain[,"energy__"])^2)
  denom <- var(chain[,"energy__"])
  numer / denom})

print(bfmi)

conditional_effects(model_IME)

# Hypothesis testing
hypothesis(model_IME, "log_area > 0")
hypothesis(model_IME, "channel_classopen > 0")

# ------------
# Custom plots
# ------------

# Create a new dataset for predictions across log_area
new_data_log_area <- data.frame(
  log_area = seq(min(data$log_area, na.rm = TRUE), 
                 max(data$log_area, na.rm = TRUE), 
                 length.out = 100),
  channel_class = "none"  # hold constant at reference level
)

# Posterior predictions
preds_log_area <- posterior_epred(model_IME, newdata = new_data_log_area)

# Summarize predictions
pred_summary_log_area <- data.frame(
  log_area = new_data_log_area$log_area,
  estimate = apply(preds_log_area, 2, mean),
  lower = apply(preds_log_area, 2, quantile, probs = 0.025),
  upper = apply(preds_log_area, 2, quantile, probs = 0.975)
)

# Plot
p_log_area <- ggplot(pred_summary_log_area, aes(x = log_area, y = estimate)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey80", alpha = 0.7) +
  geom_line(color = "red", size = 1.2) +
  geom_point(data = data, aes(x = log_area, y = log_IME_chl), 
             color = "lightblue", alpha = 0.5, size = 2) +  # show the data points
  labs(
    x = "Log(atoll area [km²])",
    y = "Log(IME magnitude [mg/m³])"
  ) +
  coord_cartesian(ylim = c(1, 6)) +
  theme_minimal(base_size = 18, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", size = 1),
    axis.text = element_text(color = "black")
  )

print(p_log_area)

# Create new data with both levels of channel_class
new_data_channel <- data.frame(
  log_area = mean(data$log_area, na.rm = TRUE),  # hold continuous predictor constant
  channel_class = factor(c("none", "open"), levels = levels(data$channel_class))
)

# Posterior predictions
preds_channel <- posterior_epred(model_IME, newdata = new_data_channel)

# Summarize predictions
pred_summary_channel <- data.frame(
  channel_class = new_data_channel$channel_class,
  estimate = apply(preds_channel, 2, mean),
  lower = apply(preds_channel, 2, quantile, probs = 0.025),
  upper = apply(preds_channel, 2, quantile, probs = 0.975)
)

# Rename factor levels for display
pred_summary_channel$channel_class <- factor(
  pred_summary_channel$channel_class,
  levels = c("none", "open"),
  labels = c("Closed", "Open")
)

# Create a new factor column in the raw data with the renamed levels
data$channel_class_label <- factor(
  data$channel_class,
  levels = c("none", "open"),
  labels = c("Closed", "Open")
)

# Plot
p_channel_class <- ggplot(pred_summary_channel, aes(x = channel_class, y = estimate)) +
  geom_point(size = 4, color = "red") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, color = "black") +
  geom_jitter(data = data, 
              aes(x = channel_class_label, y = log_IME_chl),
              width = 0.1, alpha = 0.5, color = "lightblue", size = 2) +  # use new labeled factor here
  labs(
    x = "",
    y = "log(IME magnitude [mg/m³])"
  ) +
  coord_cartesian(ylim = c(1, 6)) +
  theme_minimal(base_size = 18, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", size = 1),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(size = 18)
  )

print(p_channel_class)
