# =============================================================================
# Organ-specific frailty transitions predict mortality during human aging
# -----------------------------------------------------------------------------
# Core analysis code
# =============================================================================

# 1. LLM reasoning word clouds
# -----------------------------------------------------------------------------
# Shared analysis definitions used throughout the script.

DATA_DIR <- Sys.getenv("ORGAN_FRAILTY_DATA", unset = "data")
OUT_DIR  <- Sys.getenv("ORGAN_FRAILTY_OUT",  unset = "results")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

ORGAN_ORDER <- c(
  "cardiovascular", "hepatic", "urinary", "endocrine", "immune",
  "musculoskeletal", "neural", "gastrointestinal", "respiratory"
)


suppressPackageStartupMessages({
  library(jsonlite)
  library(tidytext)
  library(dplyr)
  library(stringr)
  library(wordcloud)
  library(RColorBrewer)
  library(ggplot2)
  library(patchwork)
  library(tidyr)
  library(purrr)
})

out_dir <- file.path(OUT_DIR, "wordclouds")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_jsonl <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  map(lines, ~ fromJSON(.x))
}

raw_list <- unlist(lapply(file.path(DATA_DIR, "reasoning"),
                          read_jsonl), recursive = FALSE)

systems <- data.frame(
  field = paste0("inference process ", 1:10),
  organ = c("overall", ORGAN_ORDER),
  stringsAsFactors = FALSE
)
systems$label <- ifelse(systems$organ == "overall",
                        "overall frailty index",
                        paste0(ORGAN_LABELS[systems$organ], " frailty index"))

extract_texts <- function(record) {
  fp <- record$frailty_prediction
  if (is.null(fp)) return(NULL)
  map_dfr(seq_len(nrow(systems)), function(i) {
    text <- fp[[systems$field[i]]]
    if (is.null(text) || is.na(text) || !nzchar(text)) return(NULL)
    data.frame(eid = as.character(record$eid), organ = systems$organ[i],
               system = systems$label[i], text = as.character(text),
               stringsAsFactors = FALSE)
  })
}

cot_df <- map_dfr(raw_list, extract_texts)

tokens_df <- cot_df %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% custom_stopwords,
         !str_detect(word, "^[0-9\\.]+$"),
         str_length(word) > 2)

freq_by_system <- tokens_df %>% count(system, word, sort = TRUE)
freq_global    <- tokens_df %>% count(word, sort = TRUE)

color_palettes <- list(
  "overall frailty index"          = brewer.pal(8, "Blues")[3:8],
  "Cardiovascular frailty index"   = brewer.pal(8, "Reds")[3:8],
  "Hepatic frailty index"          = brewer.pal(8, "Oranges")[3:8],
  "Gastrointestinal frailty index" = brewer.pal(8, "Greens")[3:8],
  "Urinary frailty index"          = brewer.pal(8, "Purples")[3:8],
  "Endocrine frailty index"        = brewer.pal(8, "YlOrBr")[3:8],
  "Musculoskeletal frailty index"  = brewer.pal(8, "GnBu")[3:8],
  "Neural frailty index"           = brewer.pal(8, "PuRd")[3:8],
  "Respiratory frailty index"      = brewer.pal(8, "BuGn")[3:8],
  "Immune frailty index"           = c("#111111", "#333333", "#555555",
                                       "#777777", "#999999", "#BBBBBB")
)

for (sys in systems$label) {
  freq_sub <- freq_by_system %>% filter(system == sys) %>%
    arrange(desc(n)) %>% slice(1:80)
  if (nrow(freq_sub) < 5) next

  pdf(file.path(out_dir, paste0("wordcloud_", which(systems$label == sys), ".pdf")),
      width = 9, height = 6)
  wordcloud(words = freq_sub$word, freq = freq_sub$n, min.freq = 1,
            max.words = 60, random.order = FALSE, rot.per = 0.2,
            colors = color_palettes[[sys]], scale = c(3.5, 0.5))
  title(main = sys, cex.main = 1.4, font.main = 2)
  dev.off()
}

pdf(file.path(out_dir, "wordcloud_overall.pdf"), width = 12, height = 8)
wordcloud(words = freq_global$word, freq = freq_global$n, min.freq = 2,
          max.words = 100, random.order = FALSE, rot.per = 0.15,
          colors = brewer.pal(9, "Set1"), scale = c(4, 0.6))
title(main = "Overall LLM reasoning across all scores", cex.main = 1.6, font.main = 2)
dev.off()

top_words_plot <- freq_by_system %>%
  group_by(system) %>% slice_max(n, n = 10) %>% ungroup() %>%
  mutate(word = reorder_within(word, n, system),
         system = factor(system, levels = systems$label))

p_bar <- ggplot(top_words_plot, aes(word, n, fill = system)) +
  geom_col(show.legend = FALSE, alpha = 0.85) +
  facet_wrap(~ system, scales = "free_y", ncol = 3) +
  coord_flip() + scale_x_reordered() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Top 10 words in LLM reasoning by score", x = NULL, y = "Frequency") +
  theme_bw(base_size = 10) +
  theme(strip.text = element_text(face = "bold", size = 8),
        plot.title = element_text(face = "bold"),
        axis.text.y = element_text(size = 8))

ggsave(file.path(out_dir, "barplot.pdf"), p_bar, width = 10, height = 12)
write.csv(freq_by_system, file.path(out_dir, "frequency_by_score.csv"), row.names = FALSE)
write.csv(freq_global, file.path(out_dir, "frequency_overall.csv"), row.names = FALSE)


# 2. surrogate models and SHAP
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(openxlsx)
  library(xgboost)
  library(shapviz)
})

COHORTS <- c("CFPS", "CHARLS", "GNHS", "NHANES")
SCORE_COLUMNS <- c("overall_frailty_index", ORGAN_ORDER)

ORGAN_COLORS <- c(
  overall_frailty_index = "#9ECAE1",
  cardiovascular        = "#FC9272",
  hepatic               = "#FDAE6B",
  gastrointestinal      = "#A1D99B",
  urinary               = "#BCBDDC",
  immune                = "#000000",
  endocrine             = "#FEC44F",
  musculoskeletal       = "#A8DDB5",
  neural                = "#C994C7",
  respiratory           = "#99D8C9"
)

out_root <- file.path(OUT_DIR, "shap")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

train_xgb_model <- function(X, y, score_name) {
  valid <- !is.na(y)
  X_fit <- X[valid, , drop = FALSE]
  y_fit <- y[valid]
  dtrain <- xgb.DMatrix(data = X_fit, label = y_fit)

  params <- list(objective = "reg:squarederror", eta = 0.05, max_depth = 6,
                 subsample = 0.8, colsample_bytree = 0.8,
                 min_child_weight = 5, nthread = 8)

  set.seed(42)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 500, nfold = 5,
               metrics = "rmse", early_stopping_rounds = 20, verbose = FALSE)
  model <- xgb.train(params = params, data = dtrain,
                     nrounds = cv$best_iteration, verbose = FALSE)
  y_pred <- predict(model, X_fit)
  r2 <- 1 - sum((y_fit - y_pred)^2) / sum((y_fit - mean(y_fit))^2)

  list(model = model, X_fit = X_fit, y_fit = y_fit, y_pred = y_pred,
       r2 = r2, cv_rmse = min(cv$evaluation_log$test_rmse_mean),
       best_nrounds = cv$best_iteration)
}

run_surrogate <- function(cohort) {
  df <- read.xlsx(file.path(DATA_DIR, "baseline", cohort),
                  check.names = FALSE)
  if (cohort == "CHARLS") {
    fi <- read.xlsx(file.path(DATA_DIR, "organ_fi", "charls"), check.names = FALSE)
    df <- inner_join(df, fi[, c("eid", SCORE_COLUMNS), drop = FALSE], by = "eid")
  }

  feature_cols <- setdiff(names(df), Filter(exclude_feature, names(df)))
  X_raw <- df[, feature_cols, drop = FALSE] %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), as.integer))
  X <- as.matrix(X_raw)

  cohort_dir <- file.path(out_root, cohort)
  dir.create(cohort_dir, recursive = TRUE, showWarnings = FALSE)
  performance <- list()

  for (sc in SCORE_COLUMNS) {
    res <- train_xgb_model(X, df[[sc]], sc)
    shp <- shapviz(res$model, X_pred = res$X_fit, X = res$X_fit)
    organ_color <- unname(ORGAN_COLORS[sc])

    p_imp <- sv_importance(shp, kind = "bar", max_display = 15, fill = organ_color) +
      labs(title = paste0("Feature importance (SHAP): ", sc),
           subtitle = sprintf("R2 = %.3f", res$r2), x = "mean(|SHAP value|)") +
      theme_bw(base_size = 12)

    fit_df <- data.frame(actual = res$y_fit, predicted = res$y_pred)
    p_scatter <- ggplot(fit_df, aes(actual, predicted)) +
      geom_point(alpha = 0.4, size = 1.2, color = organ_color) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      geom_smooth(method = "lm", se = FALSE, color = organ_color) +
      labs(title = paste0("Predicted vs actual: ", sc),
           subtitle = sprintf("R2 = %.3f, CV RMSE = %.4f (n = %d)",
                              res$r2, res$cv_rmse, length(res$y_fit)),
           x = "Actual value", y = "Predicted value") +
      theme_bw(base_size = 12)

    ggsave(file.path(cohort_dir, paste0("score_", match(sc, SCORE_COLUMNS), ".pdf")),
           p_imp + p_scatter + plot_layout(ncol = 2), width = 14, height = 6)

    imp <- data.frame(feature = colnames(shp$S),
                      importance = colMeans(abs(shp$S)),
                      mean_shap = colMeans(shp$S)) %>%
      arrange(desc(importance)) %>% mutate(rank = row_number(), score = sc)
    write.csv(imp, file.path(cohort_dir, paste0("score_", match(sc, SCORE_COLUMNS), ".csv")), row.names = FALSE)

    performance[[sc]] <- data.frame(score = sc, r2 = res$r2,
                                    cv_rmse = res$cv_rmse,
                                    best_nrounds = res$best_nrounds)
  }

  write.csv(bind_rows(performance),
            file.path(cohort_dir, "performance.csv"), row.names = FALSE)
}

for (cohort in COHORTS) run_surrogate(cohort)


# 3. DE-SWAN sliding-window trajectories
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(DEswan) })

out_root <- file.path(OUT_DIR, "deswan")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

run_deswan <- function(cohort) {
  df <- read.xlsx(file.path(DATA_DIR, "baseline", cohort),
                  check.names = FALSE)
  df$age <- as.numeric(df$age)
  df <- df[!duplicated(df$eid), ]
  rownames(df) <- df$eid

  score_cols <- intersect(SCORE_COLUMNS, names(df))
  cohort_dir <- file.path(out_root, cohort)
  dir.create(cohort_dir, recursive = TRUE, showWarnings = FALSE)

  res <- DEswan(
    data.df       = df[, score_cols, drop = FALSE],
    qt            = df$age,
    window.center = seq(30, 80, 1),
    buckets.size  = 1,
    covariates    = df[, "sex", drop = FALSE]
  )

  wide_p <- reshape.DEswan(res, parameter = 1, factor = "qt")
  write.xlsx(wide_p, file.path(cohort_dir, "pvalue.xlsx"), overwrite = TRUE)

  wide_q <- q.DEswan(wide_p, method = "BH")
  signif_q <- as.data.frame(nsignif.DEswan(wide_q))
  write.csv(signif_q, file.path(cohort_dir, "qvalue.csv"),
            row.names = TRUE)

  to_plot <- signif_q[1:3, , drop = FALSE]
  x <- as.numeric(gsub("X", "", colnames(to_plot)))
  colors <- c("#D16BA5", "#86A8E7", "#54DAD2")

  pdf(file.path(cohort_dir, "plot.pdf"), width = 5, height = 4)
  plot(1, type = "n", xlim = range(x, na.rm = TRUE), ylim = c(0, 10),
       yaxp = c(0, 10, 10), ylab = "Number of significant organ systems",
       xlab = "Age (years)")
  for (i in seq_len(nrow(to_plot))) lines(x, to_plot[i, ], lwd = i, col = colors[i])
  dev.off()
}

for (cohort in COHORTS) run_deswan(cohort)


# 4. Bayesian networks
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(bnlearn) })

out_root <- file.path(OUT_DIR, "networks")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

run_cohort_network <- function(cohort) {
  df <- read.xlsx(file.path(DATA_DIR, "baseline", cohort),
                  check.names = FALSE)
  organ_data <- df[, intersect(ORGAN_ORDER, names(df)), drop = FALSE]
  organ_data[] <- lapply(organ_data, as.numeric)
  organ_data <- as.data.frame(organ_data)

  cohort_dir <- file.path(out_root, cohort)
  dir.create(cohort_dir, recursive = TRUE, showWarnings = FALSE)

  structure <- hc(organ_data, score = "bic-g")
  boot <- boot.strength(organ_data, R = 200, algorithm = "hc",
                        algorithm.args = list(score = "bic-g"))
  avg_net <- averaged.network(boot, threshold = 0.7)
  fitted <- bn.fit(avg_net, organ_data)

  out_degree <- sapply(nodes(avg_net), function(n) length(children(avg_net, n)))

  saveRDS(structure, file.path(cohort_dir, "structure.rds"))
  saveRDS(avg_net,   file.path(cohort_dir, "network.rds"))
  saveRDS(fitted,    file.path(cohort_dir, "fit.rds"))

  sink(file.path(cohort_dir, "summary.txt"))
  cat("Cohort:", cohort, "\n")
  cat("Root nodes:", paste(root.nodes(avg_net), collapse = ", "), "\n")
  cat("Out-degree ranking:\n"); print(sort(out_degree, decreasing = TRUE))
  cat("\nNetwork structure:\n"); print(modelstring(avg_net))
  sink()
}

for (cohort in COHORTS) run_cohort_network(cohort)


# 5. Sliding-window root-node dominance
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(igraph) })

out_dir <- file.path(OUT_DIR, "sliding")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

col_cardio <- "#0072B2"; col_hepatic <- "#E69F00"

run_sliding <- function(df, organ_vars, window_size, step, min_n, age_lo, age_hi,
                        maxp = 4, absorb_lower = FALSE) {
  centers <- numeric(); n_per <- numeric()
  om <- matrix(0, 0, length(organ_vars), dimnames = list(NULL, organ_vars))
  im <- matrix(0, 0, length(organ_vars), dimnames = list(NULL, organ_vars))

  fit_window <- function(wdf, center) {
    wdf <- wdf %>% select(all_of(organ_vars)) %>% na.omit()
    if (nrow(wdf) < min_n) return(invisible(FALSE))
    net <- hc(wdf, score = "bic-g", maxp = maxp)
    g <- as.igraph(net)
    outdeg <- degree(g, mode = "out"); indeg <- degree(g, mode = "in")
    centers <<- c(centers, center); n_per <<- c(n_per, nrow(wdf))
    om <<- rbind(om, outdeg[organ_vars]); im <<- rbind(im, indeg[organ_vars] == 0)
    invisible(TRUE)
  }

  if (absorb_lower) {
    fit_window(df %>% filter(age <= age_lo + window_size / 2), age_lo)
    for (center in seq(age_lo + step, age_hi - window_size / 2, by = step))
      fit_window(df %>% filter(age >= center - window_size / 2 &
                               age <= center + window_size / 2), center)
  } else {
    for (center in seq(age_lo + window_size / 2, age_hi - window_size / 2, by = step))
      fit_window(df %>% filter(age >= center - window_size / 2 &
                               age <= center + window_size / 2), center)
  }

  long_out <- as.data.frame(om) %>% mutate(age = centers, n = n_per) %>%
    pivot_longer(all_of(organ_vars), names_to = "organ", values_to = "value") %>%
    filter(organ %in% c("cardiovascular", "hepatic"))
  long_root <- as.data.frame(im) %>% mutate(age = centers, n = n_per) %>%
    pivot_longer(all_of(organ_vars), names_to = "organ", values_to = "value") %>%
    filter(organ %in% c("cardiovascular", "hepatic"))

  list(out = long_out, root = long_root)
}

plot_sliding <- function(res, bars, title) {
  p1 <- ggplot(res$out, aes(age, value, color = organ, fill = organ)) +
    geom_point(alpha = 0.45) +
    geom_smooth(method = "loess", span = 0.75, se = TRUE, linewidth = 1.4, alpha = 0.25) +
    scale_color_manual(values = c(cardiovascular = col_cardio, hepatic = col_hepatic)) +
    scale_fill_manual(values = c(cardiovascular = col_cardio, hepatic = col_hepatic)) +
    scale_x_continuous(breaks = bars) + labs(x = NULL, y = "Out-degree") +
    theme_minimal(base_size = 13) + theme(legend.position = "top")

  p2 <- ggplot(res$root, aes(age, value, color = organ)) +
    geom_step(linewidth = 1.4) + geom_point(size = 2.6) +
    scale_color_manual(values = c(cardiovascular = col_cardio, hepatic = col_hepatic)) +
    scale_x_continuous(breaks = bars) +
    scale_y_continuous(breaks = c(0, 1), labels = c("No", "Yes"),
                       limits = c(-0.05, 1.05)) +
    labs(x = NULL, y = "Is root node") +
    theme_minimal(base_size = 13) + theme(legend.position = "none")

  d_n <- unique(res$out[, c("age", "n")]) %>% arrange(age)
  p3 <- ggplot(d_n, aes(age, n)) +
    geom_col(fill = "grey70", width = 1.4) +
    scale_x_continuous(breaks = bars) +
    labs(x = "Window centre age (years)", y = "Sample size (n)") +
    theme_minimal(base_size = 13)

  p1 / p2 / p3 + plot_annotation(title = title) +
    plot_layout(heights = c(3, 2, 1.2), axes = "collect")
}

sliding_config <- list(
  CFPS   = list(age_lo = 30, age_hi = 80, window = 5, step = 2),
  NHANES = list(age_lo = 20, age_hi = 80, window = 5, step = 2),
  CHARLS = list(age_lo = 40, age_hi = 80, window = 5, step = 2),
  GNHS   = list(age_lo = 50, age_hi = 65, window = 5, step = 2,
                absorb_lower = TRUE)
)

for (cohort in names(sliding_config)) {
  cfg <- sliding_config[[cohort]]
  df <- read.xlsx(file.path(DATA_DIR, "baseline", cohort),
                  check.names = FALSE)
  organ_vars <- intersect(ORGAN_ORDER, names(df))
  df$age <- as.numeric(df$age)

  res <- run_sliding(df, organ_vars, cfg$window, cfg$step, cfg$min_n,
                     cfg$age_lo, cfg$age_hi, absorb_lower = isTRUE(cfg$absorb_lower))
  if (nrow(res$out) == 0) next

  bars <- seq(floor(min(res$out$age) / 5) * 5,
              ceiling(max(res$out$age) / 5) * 5, by = 5)
  ggsave(file.path(out_dir, paste0("cohort_", which(names(sliding_config) == cohort), ".pdf")),
         plot_sliding(res, bars, paste0(cohort, ": shift in the dominant organ")),
         width = 8, height = 10)
}


# 6. age-stratified reference networks and transition score
# -----------------------------------------------------------------------------

AGE_YOUNG_UPPER <- 44
AGE_OLD_LOWER   <- 65

df_corrected <- read.xlsx(file.path(DATA_DIR, "organ_fi"),
                          check.names = FALSE)

df_young <- df_corrected %>% filter(age <= AGE_YOUNG_UPPER)
df_old   <- df_corrected %>% filter(age >= AGE_OLD_LOWER)

out_dir <- file.path(OUT_DIR, "reference_networks")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

learn_bn <- function(data, label) {
  mat <- data[, ORGAN_ORDER, drop = FALSE] %>%
    mutate(across(everything(), as.numeric)) %>%
    drop_na() %>% as.data.frame()

  boot <- boot.strength(mat, R = 200, algorithm = "hc",
                        algorithm.args = list(score = "bic-g"))
  avg_net <- averaged.network(boot, threshold = 0.7)
  fit <- bn.fit(avg_net, mat, method = "mle")

  list(network = avg_net, fitted = fit, roots = root.nodes(avg_net), data = mat,
       out_degree = sapply(nodes(avg_net),
                           function(n) length(children(avg_net, n))))
}

bn_young <- learn_bn(df_young, "Young")
bn_old   <- learn_bn(df_old,   "Older")

plot_one_network <- function(bn_result, title_str) {
  arcs_df <- as.data.frame(arcs(bn_result$network))
  g <- graph_from_data_frame(arcs_df, directed = TRUE, vertices = ORGAN_ORDER)

  in_deg  <- degree(g, mode = "in")
  out_deg <- degree(g, mode = "out")
  roots <- names(in_deg[in_deg == 0 & out_deg > 0])

  plot(g,
       vertex.color = ifelse(V(g)$name %in% roots, "#E74C3C", "#3498DB"),
       vertex.frame.color = ifelse(V(g)$name %in% roots, "#E74C3C", "#BDC3C7"),
       vertex.size = ifelse(V(g)$name %in% roots, 38, 26),
       vertex.label = toupper(V(g)$name), vertex.label.cex = 0.8,
       vertex.label.font = ifelse(V(g)$name %in% roots, 2, 1),
       edge.arrow.size = 0.45, edge.color = "#7F8C8D",
       layout = layout_with_sugiyama(g)$layout, main = title_str)
}

pdf(file.path(out_dir, "networks.pdf"), width = 16, height = 8)
par(mfrow = c(1, 2))
plot_one_network(bn_young, "Young reference network")
plot_one_network(bn_old,   "Older reference network")
dev.off()

# Per-organ conditional log-likelihood
organ_conditional_loglik <- function(fitted_bn, data, node) {
  nd  <- fitted_bn[[node]]
  par <- nd$parents
  cf  <- nd$coefficients

  if (length(par) == 0) {
    mu <- rep(unname(cf["(Intercept)"]), nrow(data))
  } else {
    mu <- unname(cf["(Intercept)"]) +
      as.numeric(as.matrix(data[, par, drop = FALSE]) %*% cf[par])
  }
  dnorm(data[[node]], mean = mu, sd = nd$sd, log = TRUE)
}

conditional_loglik_matrix <- function(fitted_bn, data) {
  out <- sapply(ORGAN_ORDER, function(node)
    organ_conditional_loglik(fitted_bn, data, node))
  colnames(out) <- ORGAN_ORDER
  out
}

compute_transition_score <- function(cohort_data) {
  dat <- cohort_data[, ORGAN_ORDER, drop = FALSE] %>%
    mutate(across(everything(), as.numeric)) %>% as.data.frame()

  idx <- which(complete.cases(dat))
  dat_c <- dat[idx, , drop = FALSE]

  ll_young <- conditional_loglik_matrix(bn_young$fitted, dat_c)
  ll_old   <- conditional_loglik_matrix(bn_old$fitted,   dat_c)

  organ_score <- ll_old - ll_young
  colnames(organ_score) <- paste0("transition_score_", ORGAN_ORDER)

  out <- matrix(NA_real_, nrow = nrow(dat), ncol = length(ORGAN_ORDER),
                dimnames = list(NULL, colnames(organ_score)))
  out[idx, ] <- organ_score

  data.frame(eid = cohort_data$eid, out, check.names = FALSE,
             transition_score = NA_real_)
}

ts_res <- compute_transition_score(df)
ts_res$transition_score <- rowSums(ts_res[, paste0("transition_score_", ORGAN_ORDER)])

df_ts <- merge(df, ts_res, by = "eid") 

write.xlsx(df_ts, file.path(out_dir, "transition_score.xlsx"),
           overwrite = TRUE)


# 7. transition-score landscape and mortality
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ppcor)
  library(scales)
  library(png)
  library(pdftools)
})

COL_ADVERSE <- "#C0392B"
COL_PROTECT <- "#2E6DA4"
COL_MID     <- "#E67E22"
COL_INK     <- "#2C3E50"

theme_nm <- function(base_size = 8) {
  ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      text              = ggplot2::element_text(colour = "black"),
      plot.title        = ggplot2::element_text(size = ggplot2::rel(1.15), face = "bold",
                                                colour = COL_INK, hjust = 0,
                                                margin = ggplot2::margin(b = 2)),
      plot.subtitle     = ggplot2::element_text(size = ggplot2::rel(0.85), colour = "grey40",
                                                hjust = 0, margin = ggplot2::margin(b = 6)),
      axis.title        = ggplot2::element_text(size = ggplot2::rel(1.0), colour = COL_INK),
      axis.text         = ggplot2::element_text(size = ggplot2::rel(0.85), colour = "#333333"),
      axis.line         = ggplot2::element_line(colour = "#444444", linewidth = 0.4),
      axis.ticks        = ggplot2::element_line(colour = "#444444", linewidth = 0.4),
      legend.title      = ggplot2::element_text(size = ggplot2::rel(0.85), colour = COL_INK),
      legend.text       = ggplot2::element_text(size = ggplot2::rel(0.85)),
      legend.background = ggplot2::element_blank(),
      legend.key        = ggplot2::element_blank(),
      strip.text        = ggplot2::element_text(size = ggplot2::rel(0.9), face = "bold",
                                                colour = COL_INK),
      strip.background  = ggplot2::element_blank(),
      panel.border      = ggplot2::element_blank(),
      panel.grid        = ggplot2::element_blank(),
      panel.background  = ggplot2::element_rect(fill = "white", colour = NA),
      plot.background   = ggplot2::element_rect(fill = "white", colour = NA),
      plot.margin       = ggplot2::margin(4, 6, 4, 4)
    )
}

panel_tag <- function(plot, tag) {
  cowplot::ggdraw(plot) +
    cowplot::draw_label(tag, x = 0.004, y = 0.995, hjust = 0, vjust = 1,
                        fontface = "bold", size = 13, colour = "black")
}

out_dir <- file.path(OUT_DIR, "transition_score")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

render_pdf_to_png <- function(pdf_path, png_path, dpi = 300) {
  if (file.exists(png_path)) return(TRUE)
  if (!file.exists(pdf_path)) return(FALSE)
  tryCatch({
    img <- pdftools::pdf_render_page(pdf_path, page = 1, dpi = dpi)
    png::writePNG(img, png_path)
    file.exists(png_path)
  }, error = function(e) FALSE)
}

gdf <- df_ts
gdf$sexn <- ifelse(grepl("^1\\.|^man$|^m$|^male$", tolower(trimws(as.character(gdf$sex)))), 1,
            ifelse(grepl("^2\\.|^woman$|^f$|^female$", tolower(trimws(as.character(gdf$sex)))), 0, NA))

score01 <- function(x) {
  p1 <- quantile(x, 0.01, na.rm = TRUE); p99 <- quantile(x, 0.99, na.rm = TRUE)
  w <- pmin(pmax(x, p1), p99); (w - p1) / (p99 - p1)
}

# Panel a: reference networks
net_png <- file.path(out_dir, "networks.png")
render_pdf_to_png(file.path(OUT_DIR, "reference_networks",
                            "networks.pdf"), net_png, dpi = 300)
p_a <- if (file.exists(net_png)) {
  cowplot::ggdraw() + cowplot::draw_image(net_png) +
    cowplot::draw_label("a", x = 0.004, y = 0.995, hjust = 0, vjust = 1,
                        fontface = "bold", size = 13)
} else {
  ggplot() + theme_void()
}

# Panel b: per-organ score versus overall score
rho_df <- do.call(rbind, lapply(ORGAN_ORDER, function(o) {
  m <- na.omit(data.frame(x = gdf[[paste0("transition_score_", o)]],
                          y = gdf$transition_score, age = gdf$age, sex = gdf$sexn))
  cor_ <- pcor(m, method = "spearman")
  data.frame(organ = o, rho = cor_$estimate["x", "y"],
             p = cor_$p.value["x", "y"])
}))
rho_df$label <- factor(ORGAN_LABELS[rho_df$organ],
                       levels = ORGAN_LABELS[rho_df$organ[order(rho_df$rho)]])

p_b <- ggplot(rho_df, aes(rho, label)) +
  geom_vline(xintercept = 0, colour = "grey65", linewidth = 0.4) +
  geom_segment(aes(xend = 0, yend = label), colour = "grey75", linewidth = 0.6) +
  geom_point(aes(colour = rho), size = 4.2) +
  geom_text(aes(label = sprintf("%.2f", rho)), hjust = -0.4, size = 2.8,
            colour = "grey25") +
  scale_colour_gradient(low = "#C8DCF0", high = COL_PROTECT, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Per-organ vs overall transition score",
       subtitle = "Partial Spearman, adjusted for age and sex",
       x = expression(Partial~Spearman~rho), y = NULL) +
  theme_nm(9)

# Panel c: age versus score
d_age <- gdf %>% filter(!is.na(age), !is.na(transition_score), !is.na(sexn))
d_age$score01 <- score01(d_age$transition_score)
q1 <- quantile(d_age$score01, 1/3); q2 <- quantile(d_age$score01, 2/3)

d_age$tertile <- factor(ifelse(d_age$score01 <= q1, "T1 (low)",
                               ifelse(d_age$score01 <= q2, "T2 (mid)", "T3 (high)")),
                        levels = c("T1 (low)", "T2 (mid)", "T3 (high)"))
d_age$age_group <- ifelse(d_age$age < 60, "Young (<60)", "Old (>=60)")
d_age$cell <- paste(d_age$age_group, d_age$tertile)

cell_cols <- c("Young (<60) T1 (low)" = "#C6DBEF", "Young (<60) T2 (mid)" = "#F6E0BE",
               "Young (<60) T3 (high)" = COL_ADVERSE, "Old (>=60) T1 (low)" = COL_PROTECT,
               "Old (>=60) T2 (mid)" = "#E8D5C0", "Old (>=60) T3 (high)" = "#E8B8B2")

cell_pct <- d_age %>% count(cell) %>% mutate(pct = n / sum(n) * 100) %>%
  left_join(data.frame(cell = names(cell_cols),
                       x = c(28, 28, 28, 88, 88, 88),
                       ypos = c(q1/2, (q1+q2)/2, (q2+1)/2,
                                q1/2, (q1+q2)/2, (q2+1)/2)),
            by = "cell") %>% mutate(lab = sprintf("%.1f%%", pct))

p_c <- ggplot(d_age, aes(age, score01)) +
  geom_point(aes(colour = cell), alpha = 0.10, size = 0.4) +
  geom_vline(xintercept = 60, linetype = "dashed", colour = "grey40",
             linewidth = 0.5) +
  geom_hline(yintercept = c(q1, q2), linetype = "dashed", colour = "grey40",
             linewidth = 0.5, alpha = 0.8) +
  geom_label(data = cell_pct, aes(x = x, y = ypos, label = lab, fill = cell),
             colour = "white", size = 2.6, fontface = "bold", label.size = 0.25,
             inherit.aes = FALSE) +
  scale_fill_manual(values = cell_cols, guide = "none") +
  scale_colour_manual(values = cell_cols, name = NULL,
                      guide = guide_legend(override.aes = list(alpha = 1, size = 3))) +
  coord_cartesian(xlim = c(min(d_age$age), 110), ylim = c(0, 1)) +
  labs(title = "Age vs transition score",
       subtitle = "Winsorised (1%/99%) and scaled to [0,1]",
       x = "Age (years)", y = "Normalised transition score") +
  theme_nm(9) + theme(legend.position = "bottom",
                      legend.text = element_text(size = 7.5))

# Panel d: disease prevalence (pre-rendered panel)
prevalence_panel <- file.path(DATA_DIR, "prevalence")
prev_png <- file.path(out_dir, "prevalence.png")
render_pdf_to_png(prevalence_panel, prev_png, dpi = 300)
p_d <- if (file.exists(prev_png)) {
  cowplot::ggdraw() + cowplot::draw_image(prev_png) +
    cowplot::draw_label("d", x = 0.004, y = 0.995, hjust = 0, vjust = 1,
                        fontface = "bold", size = 13)
} else {
  ggplot() + theme_void()
}

# Panel e: mortality by score tertile
cox <- read.csv(file.path(DATA_DIR, "tertile_cox"),
                stringsAsFactors = FALSE, check.names = FALSE) %>%
  filter(grepl("adjust", model, ignore.case = TRUE)) %>%
  mutate(HR = as.numeric(HR), lo = as.numeric(`HR.confint.lower.95`),
         hi = as.numeric(`HR.confint.upper.95`), pval = as.numeric(pval),
         sig = ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**",
                    ifelse(pval < 0.05, "*", "ns"))),
         tertile = ifelse(grepl("T2", comparison), "T2 (medium)", "T3 (high)")) %>%
  filter(tertile %in% c("T2 (medium)", "T3 (high)"))

cox$cohort <- factor(cox$cohort, levels = rev(c("CFPS", "CHARLS", "GNHS", "NHANES")))
cox$ypos <- as.numeric(cox$cohort) +
  ifelse(cox$tertile == "T2 (medium)", -0.22, 0.22)

p_e <- ggplot(cox, aes(HR, ypos)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey55",
             linewidth = 0.5) +
  geom_errorbarh(aes(xmin = lo, xmax = hi, colour = tertile), height = 0.16,
                 linewidth = 0.9) +
  geom_point(aes(colour = tertile), size = 3.2) +
  geom_text(aes(label = paste0(`HR (95% CI)`, sig)), hjust = -0.12,
            size = 2.4, colour = "grey25") +
  scale_x_log10(expand = expansion(mult = c(0.02, 0.18))) +
  scale_y_continuous(breaks = 1:4, labels = levels(cox$cohort),
                     limits = c(0.6, 4.4)) +
  scale_colour_manual(values = c("T2 (medium)" = COL_MID, "T3 (high)" = COL_ADVERSE),
                      name = "Transition-score tertile") +
  labs(title = "Transition-score tertile and mortality",
       subtitle = "Adjusted for age and sex; reference = T1 (low)",
       x = "Hazard ratio (95% CI)", y = NULL) +
  theme_nm(9) + theme(legend.position = "top")

# Panel f: frailty-index mortality within each tertile
fi_eff <- read.csv(file.path(DATA_DIR, "tertile_fi"),
                   stringsAsFactors = FALSE, check.names = FALSE) %>%
  mutate(HR = as.numeric(HR), lo = as.numeric(`HR.confint.lower.95`),
         hi = as.numeric(`HR.confint.upper.95`), pval = as.numeric(pval),
         sig = ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**",
                    ifelse(pval < 0.05, "*", "ns"))),
         lab = paste0(`HR (95% CI)`, sig),
         tertile = factor(ifelse(grepl("T1", group), "Low (T1)",
                          ifelse(grepl("T2", group), "Medium (T2)", "High (T3)")),
                          levels = c("Low (T1)", "Medium (T2)", "High (T3)")))

fi_eff$cohort <- factor(fi_eff$cohort,
                        levels = c("CFPS", "CHARLS", "GNHS", "NHANES"))
tert_cols <- c("Low (T1)" = COL_PROTECT, "Medium (T2)" = COL_MID,
               "High (T3)" = COL_ADVERSE)

p_f <- ggplot(fi_eff, aes(HR, tertile)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey55",
             linewidth = 0.5) +
  geom_errorbarh(aes(xmin = lo, xmax = hi, colour = tertile), height = 0.16,
                 linewidth = 0.9) +
  geom_point(aes(fill = tertile), size = 3.4, shape = 21, stroke = 0.7,
             colour = "white") +
  geom_text(aes(label = lab), hjust = -0.15, size = 2.6, vjust = -1.0,
            colour = "grey25") +
  scale_colour_manual(values = tert_cols, guide = "none") +
  scale_fill_manual(values = tert_cols, name = "Transition-score tertile") +
  scale_x_log10(limits = c(0.4, 2.0),
                breaks = c(0.4, 0.6, 1.0, 1.5, 2.0)) +
  facet_wrap(~ cohort, ncol = 4) +
  labs(title = "Frailty index and mortality within each transition-score tertile",
       subtitle = "HR of frailty index (per 1 SD) on mortality; adjusted for age and sex",
       x = "Hazard ratio (95% CI, log scale)", y = NULL) +
  theme_nm(9) + theme(strip.text = element_text(face = "bold"),
                      panel.spacing = unit(8, "pt"),
                      legend.position = "bottom")

transition_plot <- (p_a + panel_tag(p_b, "b") + panel_tag(p_c, "c") +
           plot_layout(ncol = 3, widths = c(1.0, 0.9, 1.25))) /
  (p_d + panel_tag(p_e, "e") + plot_layout(ncol = 2)) /
  panel_tag(p_f, "f") +
  plot_layout(heights = c(100, 96, 64))

ggsave(file.path(out_dir, "main.pdf"), transition_plot,
       width = 210, height = 300, units = "mm", limitsize = FALSE)
ggsave(file.path(out_dir, "main.png"), transition_plot,
       width = 210, height = 300, units = "mm", dpi = 300, limitsize = FALSE)


# 8. disease burden, lifestyle and the transition score
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(readr) })

base_dir <- file.path(DATA_DIR, "contribution")

DISEASE_COLUMNS <- list(
  CHARLS = c("hypertension", "dyslipidemia", "diabetes", "cancer", "lung_disease",
             "liver_disease", "heart_disease", "stroke", "kidney_disease",
             "stomach_disease", "arthritis", "asthma"),
  CFPS   = c("chronic_disease"),
  GNHS   = c("htn", "dm", "dyslipidemia", "heart_attack", "stroke", "chd",
             "cancer", "asthma", "hepatitis", "gout", "ra", "depression",
             "gastric_ulcer", "insomnia", "bronchitis", "anemia")
)

DISEASE_LABELS <- list(
  CHARLS = c("Hypertension", "Dyslipidemia", "Diabetes", "Cancer", "Lung disease",
             "Liver disease", "Heart disease", "Stroke", "Kidney disease",
             "Stomach disease", "Arthritis", "Asthma"),
  CFPS   = c("Chronic disease"),
  GNHS   = c("Hypertension", "Diabetes", "Dyslipidemia", "Heart attack", "Stroke",
             "CHD", "Cancer", "Asthma", "Hepatitis", "Gout", "RA", "Depression",
             "Gastric ulcer", "Insomnia", "Bronchitis", "Anemia",
             "HTN medication", "DM medication", "Dyslipidemia medication")
)

fit_comorbidity <- function(data, disease_cols, id_var) {
  dat <- data.frame(
    transition_score = data$transition_score,
    age = data$age, sex = data$sex,
    comorbidity = rowSums(data[, disease_cols, drop = FALSE], na.rm = TRUE),
    id = data[[id_var]]
  ) %>% filter(!is.na(transition_score), !is.na(age), !is.na(sex))
  dat$comorbidity <- as.numeric(scale(dat$comorbidity))

  model <- NULL
  if (mean(table(dat$id) > 1) > 0.01 && length(unique(dat$id)) > 50) {
    model <- tryCatch(lmer(transition_score ~ comorbidity + age + sex + (1 | id),
                           data = dat, REML = FALSE), error = function(e) NULL)
  }
  if (is.null(model)) {
    model <- lm(transition_score ~ comorbidity + age + sex, data = dat)
  }

  s <- as.data.frame(coef(summary(model)))
  s$term <- rownames(s)
  data.frame(label = "Diseases (comorbidity)",
             estimate = s$Estimate[s$term == "comorbidity"],
             se = s$`Std. Error`[s$term == "comorbidity"])
}

build_contribution <- function(cohort) {
  dis_data <- read_csv(file.path(base_dir, cohort, "data"),
                       show_col_types = FALSE)
  id_var <- if ("ID" %in% names(dis_data)) "ID" else "eid"
  comorbid <- fit_comorbidity(dis_data, DISEASE_COLUMNS[[cohort]], id_var)

  res <- read.csv(file.path(base_dir, cohort, "results"),
                  stringsAsFactors = FALSE) %>%
    filter(!(label %in% DISEASE_LABELS[[cohort]]))

  out <- rbind(comorbid, res[, c("label", "estimate", "se")])
  out$category <- c("Disease", rep("Lifestyle", nrow(res)))
  out$label <- factor(out$label, levels = out$label[order(out$estimate)])
  out
}

plot_contribution <- function(dd, title) {
  dd$direction <- ifelse(dd$category == "Disease", "Diseases / adverse",
                         ifelse(dd$estimate >= 0, "Adverse lifestyle",
                                "Healthy lifestyle"))
  xlim <- max(abs(dd$estimate)) * 1.30
  ggplot(dd, aes(estimate, label, fill = direction)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = estimate - 1.96 * se, xmax = estimate + 1.96 * se),
                   height = 0.2, colour = "grey50", linewidth = 0.5) +
    geom_col(width = 0.62, alpha = 0.92) +
    scale_fill_manual(values = c("Diseases / adverse" = COL_ADVERSE,
                                 "Adverse lifestyle" = "#D97A6B",
                                 "Healthy lifestyle" = COL_PROTECT), name = NULL) +
    coord_cartesian(xlim = c(-xlim, xlim)) +
    labs(title = title, x = "Beta per additional disease or per SD of factor") +
    theme_nm(9) + theme(legend.position = "bottom",
                        plot.title = element_text(size = 10, face = "bold"))
}

p_d <- plot_contribution(build_contribution("CFPS"),   "CFPS")
p_e <- plot_contribution(build_contribution("CHARLS"), "CHARLS")
p_f <- plot_contribution(build_contribution("GNHS"),   "GNHS")


# 9. individual transition trajectories
# -----------------------------------------------------------------------------

cdf <- read.csv(file.path(DATA_DIR, "cases"),
                stringsAsFactors = FALSE)

CASE_META <- list(
  list(id = "CASE_A", direction = "down", event_year = 2012,
       high_year = 2010, high_event = "active liver disease",
       low_year = 2012, low_event = "liver disease resolved",
       title = "Case A: liver disease resolved"),
  list(id = "CASE_B", direction = "down", event_year = 2016,
       high_year = 2010, high_event = "heavy drinker",
       low_year = 2016, low_event = "quit drinking",
       title = "Case B: quit alcohol, daily exercise"),
  list(id = "CASE_C", direction = "down", event_year = 2018,
       high_year = 2012, high_event = "smoker + drinker",
       low_year = 2018, low_event = "quit both",
       title = "Case C: quit smoking and drinking"),
  list(id = "CASE_D", direction = "up", event_year = 2018,
       high_year = 2018, high_event = "diabetes + bronchitis",
       low_year = 2010, low_event = "smoker but healthy",
       title = "Case D: new diseases after healthy period"),
  list(id = "CASE_E", direction = "up", event_year = 2022,
       high_year = 2022, high_event = "new diabetes",
       low_year = 2012, low_event = "healthy young woman",
       title = "Case E: new diabetes in young woman"),
  list(id = "CASE_F", direction = "up", event_year = 2018,
       high_year = 2018, high_event = "stroke + hyperlipidemia",
       low_year = 2012, low_event = "healthy, non-smoker",
       title = "Case F: stroke and hyperlipidemia")
)

make_case <- function(meta) {
  person <- cdf %>% filter(ID == meta$id) %>% arrange(year) %>%
    mutate(score01 = score01(transition_score))
  if (nrow(person) < 2) return(NULL)

  down <- meta$direction == "down"
  before_col <- if (down) COL_ADVERSE else COL_PROTECT
  after_col  <- if (down) COL_PROTECT else COL_ADVERSE

  seg <- data.frame(x = person$year[-nrow(person)], xend = person$year[-1],
                    y = person$score01[-nrow(person)], yend = person$score01[-1])
  seg$colour <- ifelse(seg$xend >= meta$event_year, after_col, before_col)
  seg$colour[seg$x >= meta$event_year] <- after_col

  high_val <- person$score01[match(meta$high_year, person$year)]
  low_val  <- person$score01[match(meta$low_year, person$year)]

  p <- ggplot(person, aes(year, score01)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
               linewidth = 0.4) +
    annotate("rect", xmin = min(person$year) - 1, xmax = meta$event_year - 0.3,
             ymin = -Inf, ymax = Inf, fill = before_col, alpha = 0.05) +
    annotate("rect", xmin = meta$event_year - 0.3, xmax = max(person$year) + 1,
             ymin = -Inf, ymax = Inf, fill = after_col, alpha = 0.05) +
    geom_segment(data = seg, aes(x, xend = xend, y = y, yend = yend),
                 colour = seg$colour, linewidth = 1.6) +
    geom_point(aes(fill = ifelse(year >= meta$event_year, after_col, before_col)),
               size = 3.2, shape = 21, colour = COL_INK, stroke = 0.8) +
    scale_fill_identity() +
    geom_text(aes(label = sprintf("%.2f", score01),
                  vjust = ifelse(score01 >= 0.5, -1.5, 2.2)),
              size = 2.5, fontface = "bold", colour = COL_INK) +
    labs(title = meta$title, x = "Survey year", y = "Normalised transition score") +
    theme_nm(8) + coord_cartesian(ylim = c(0, 1.08))

  high_txt <- sprintf("%d: %s (score %.2f)", meta$high_year,
                      meta$high_event, high_val)
  low_txt  <- sprintf("%d: %s (score %.2f)", meta$low_year,
                      meta$low_event, low_val)

  if (down) {
    p + annotate("label", x = meta$high_year, y = 0.90, label = high_txt,
                 hjust = -0.02, size = 2.4, fontface = "bold",
                 fill = "#FDECEA", colour = COL_ADVERSE, label.size = 0.4) +
      annotate("label", x = meta$low_year, y = 0.10, label = low_txt,
               size = 2.4, fontface = "bold", fill = "white", colour = COL_PROTECT)
  } else {
    p + annotate("label", x = meta$low_year, y = 0.10, label = low_txt,
                 size = 2.4, fontface = "bold", fill = "#EAF2FB",
                 colour = COL_PROTECT) +
      annotate("label", x = meta$high_year, y = 0.90, label = high_txt,
               size = 2.4, fontface = "bold", fill = "white", colour = COL_ADVERSE)
  }
}

case_plots <- Filter(Negate(is.null), lapply(CASE_META, make_case))

out_dir <- file.path(OUT_DIR, "transition_dynamics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dynamics_plot <- panel_tag(p_d, "d") + panel_tag(p_e, "e") + panel_tag(p_f, "f") +
  plot_layout(ncol = 3)
if (length(case_plots) >= 3) {
  dynamics_plot <- dynamics_plot / (panel_tag(case_plots[[1]], "g") +
                    panel_tag(case_plots[[2]], "h") +
                    panel_tag(case_plots[[3]], "i") + plot_layout(ncol = 3))
}
if (length(case_plots) >= 6) {
  dynamics_plot <- dynamics_plot / (panel_tag(case_plots[[4]], "j") +
                    panel_tag(case_plots[[5]], "k") +
                    panel_tag(case_plots[[6]], "l") + plot_layout(ncol = 3))
}

ggsave(file.path(out_dir, "main.pdf"), dynamics_plot,
       width = 210, height = 300, units = "mm", limitsize = FALSE)


# 10. proteomic landscape
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(ggrepel) })

ORGAN_ORDER_PROTEOMICS <- c(
  "endocrine", "hepatic", "cardiovascular", "urinary", "immune",
  "neural", "musculoskeletal", "gastrointestinal", "respiratory"
)

organ_labels_short <- c(
  endocrine = "Endocrine", hepatic = "Hepatic",
  cardiovascular = "Cardiovascular", urinary = "Urinary", immune = "Immune",
  neural = "Neural", musculoskeletal = "Musculoskeletal",
  gastrointestinal = "Gastrointestinal", respiratory = "Respiratory"
)

out_dir <- file.path(OUT_DIR, "proteomics_landscape")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df_prot <- read.csv(file.path(DATA_DIR, "protein_organ"),
                    stringsAsFactors = FALSE)
df_sig  <- df_prot[df_prot$fdr < 0.05, ]

theme_mini <- function(base_size = 7) {
  theme_minimal(base_size = base_size) %+replace%
    theme(text = element_text(family = "sans"),
          axis.title = element_text(size = 8),
          axis.text = element_text(size = 7, color = "black"),
          axis.ticks = element_line(color = "black", linewidth = 0.3),
          axis.line = element_line(color = "black", linewidth = 0.4),
          panel.grid = element_blank(), panel.border = element_blank(),
          plot.subtitle = element_text(size = 7, hjust = 0, margin = margin(b = 4)))
}

count_df <- df_sig %>% count(organ, name = "n_total")
count_df$organ <- factor(count_df$organ,
                         levels = count_df$organ[order(count_df$n_total)])

p_count <- ggplot(count_df, aes(organ, n_total, fill = organ)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.72) +
  geom_text(aes(label = n_total), hjust = -0.25, size = 2.6, fontface = "bold") +
  scale_fill_manual(values = ORGAN_COLORS, guide = "none") +
  scale_x_discrete(labels = organ_labels_short) +
  coord_flip() +
  labs(subtitle = "Significant proteins by organ system",
       x = NULL, y = "No. of significant proteins") +
  theme_mini()

share <- table(sapply(split(df_sig$organ, df_sig$protein),
                      function(x) length(unique(x))))
share_df <- data.frame(n_organ = as.numeric(names(share)), count = as.numeric(share))

p_share <- ggplot(share_df, aes(n_organ, count)) +
  geom_bar(stat = "identity", fill = "#5A5A5A", color = "black",
           linewidth = 0.35, width = 0.6) +
  geom_text(aes(label = count), vjust = -0.4, size = 2.6, fontface = "bold") +
  labs(subtitle = "Protein sharing across organ systems",
       x = "No. of organ systems", y = "No. of proteins") +
  theme_mini()

df_prot$neg_log10_pval <- -log10(pmax(df_prot$pval, 1e-300))
df_prot$sig <- df_prot$fdr < 0.05

top10 <- df_sig %>% group_by(protein, gene) %>%
  summarise(beta = beta[which.min(pval)],
            neg_log10_pval = max(-log10(pmax(pval, 1e-300))),
            .groups = "drop") %>%
  arrange(desc(neg_log10_pval)) %>% head(10)

p_volcano <- ggplot(df_prot, aes(beta, neg_log10_pval, color = organ,
                                 alpha = sig, size = sig)) +
  geom_point(stroke = 0.15) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed",
             color = "grey50", linewidth = 0.3) +
  scale_color_manual(values = ORGAN_COLORS, guide = "none") +
  scale_alpha_manual(values = c(`TRUE` = 0.9, `FALSE` = 0.2), guide = "none") +
  scale_size_manual(values = c(`TRUE` = 1.8, `FALSE` = 0.9), guide = "none") +
  geom_text_repel(data = top10, aes(beta, neg_log10_pval, label = gene),
                  inherit.aes = FALSE, size = 2.5, fontface = "bold",
                  max.overlaps = 20, min.segment.length = 0) +
  labs(subtitle = "Effect size vs significance",
       x = expression(beta ~ coefficient),
       y = expression(-log[10](italic(p)))) +
  theme_mini()

ggsave(file.path(out_dir, "main.pdf"),
       p_count + p_share + p_volcano + plot_layout(ncol = 3),
       width = 210, height = 75, units = "mm")


# 11. protein score
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(survival); library(colorspace); library(readxl); library(ggraph)
  library(tidygraph); library(ggforce)
})

col_blue <- "#2166AC"; col_red <- "#D7191C"

out_dir <- file.path(OUT_DIR, "protein_score")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

shade_fill <- function(organ, dark = TRUE) {
  col <- unname(ORGAN_COLORS[organ])
  if (identical(col, "#000000")) return(if (dark) "#000000" else "#C0C0C0")
  if (dark) darken(col, amount = 0.35) else lighten(col, amount = 0.45)
}

# Panel 1: external validation
val_sum <- read.csv(file.path(DATA_DIR, "validation_summary"),
                    stringsAsFactors = FALSE)
plot1_df <- val_sum %>% filter(organ %in% ORGAN_ORDER_PROTEOMICS) %>%
  mutate(not_validated = total - validated,
         organ = factor(organ, levels = rev(ORGAN_ORDER_PROTEOMICS))) %>%
  pivot_longer(c(validated, not_validated), names_to = "type", values_to = "n") %>%
  mutate(grp = paste(organ, type, sep = "_"))

plot1_fill <- c()
for (o in levels(plot1_df$organ)) {
  plot1_fill[paste(o, "validated", sep = "_")]     <- shade_fill(o, TRUE)
  plot1_fill[paste(o, "not_validated", sep = "_")] <- shade_fill(o, FALSE)
}

p1 <- ggplot(plot1_df, aes(organ, n, fill = grp)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.72,
           position = position_stack(reverse = TRUE)) +
  scale_fill_manual(values = plot1_fill, guide = "none") +
  scale_x_discrete(labels = organ_labels_short) +
  coord_flip() +
  labs(subtitle = "External validation: validated (dark) vs not validated (light)",
       x = NULL, y = "No. of proteins") +
  theme_nm(8)

# Panel 2: elastic-net coefficients
sel <- read.csv(file.path(DATA_DIR, "selected_proteins"),
                stringsAsFactors = FALSE)
idmap <- read.delim(file.path(DATA_DIR, "mapping"),
                    stringsAsFactors = FALSE)[, 1:2]
names(idmap) <- c("From", "To"); idmap$From <- tolower(idmap$From)
sel$gene <- idmap$To[match(tolower(sel$protein), idmap$From)]
sel$gene[is.na(sel$gene)] <- sel$protein[is.na(sel$gene)]
sel <- sel %>% arrange(coefficient) %>% mutate(gene = factor(gene, levels = gene))

p2 <- ggplot(sel, aes(gene, coefficient, fill = coefficient > 0)) +
  geom_col(width = 0.68, color = "black", linewidth = 0.22) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.4) +
  scale_fill_manual(values = c(`TRUE` = col_red, `FALSE` = col_blue),
                    guide = "none") +
  coord_flip() +
  labs(subtitle = "Elastic-net selected proteins", x = NULL,
       y = "Elastic-net coefficient") +
  theme_nm(7.2) + theme(axis.text.y = element_text(size = 5.2))

# Discovery data
df_full <- read.csv(file.path(DATA_DIR, "proteomics"),
                    stringsAsFactors = FALSE)
df_full <- df_full[order(df_full$ID, df_full$time_x), ]
df_full$time <- unlist(tapply(seq_len(nrow(df_full)), df_full$ID, seq_along))
df_first <- df_full[df_full$time == 1, ]; names(df_first)[1] <- "eid"

ts <- read.csv(file.path(DATA_DIR, "longitudinal"),
               stringsAsFactors = FALSE)
if ("cohort" %in% names(ts)) ts <- ts[ts$cohort == "GNHS", ]
ts <- ts[, c("eid", "transition_score")]

score <- read.csv(file.path(DATA_DIR, "score_discovery"),
                  stringsAsFactors = FALSE)

df3 <- merge(merge(df_first, ts, by = "eid"), score, by = "eid")
df3$transition_score_z <- as.numeric(scale(df3$transition_score))
rho3 <- cor(df3$protein_score, df3$transition_score_z, method = "spearman")
df3p <- df3[df3$transition_score_z <= quantile(df3$transition_score_z, 0.99,
                                               na.rm = TRUE), ]

p3 <- ggplot(df3p, aes(protein_score, transition_score_z)) +
  geom_point(alpha = 0.40, size = 1.9, shape = 21, fill = col_blue, stroke = 0.4) +
  geom_smooth(method = "lm", color = col_red, fill = alpha(col_red, 0.18),
              linewidth = 0.8, se = TRUE) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.6,
           label = sprintf("Spearman rho = %.3f (P < 0.001)", rho3),
           size = 2.8) +
  labs(subtitle = "Protein score vs transition score (discovery)",
       x = "Protein score", y = "Transition score (z)") +
  theme_nm(8)

# Panel 4: cumulative mortality
df4 <- merge(merge(df_first, ts, by = "eid"), score, by = "eid")
df_full$age <- as.numeric(df_full$age)
max_age <- aggregate(age ~ ID, data = df_full, FUN = max, na.rm = TRUE)
names(max_age)[2] <- "max_age"
df4 <- merge(df4, max_age, by = "ID", all.x = TRUE)
df4$surv_time <- ifelse(df4$mortality == 1, df4$death_age - df4$age,
                        df4$max_age - df4$age)

cox_data <- df4[, c("surv_time", "mortality", "protein_score", "age", "sex")]
cox_data <- cox_data[complete.cases(cox_data), ]
cut_points <- quantile(cox_data$protein_score, c(0, 1/3, 2/3, 1), na.rm = TRUE)
cox_data$group <- cut(cox_data$protein_score, breaks = cut_points,
                      labels = c("Low", "Medium", "High"), include.lowest = TRUE)
km <- survfit(Surv(surv_time, mortality) ~ group, data = cox_data)

km_sum <- summary(km)
km_df <- data.frame(time = km_sum$time, inc = 1 - km_sum$surv,
                    lower = 1 - km_sum$upper, upper = 1 - km_sum$lower)
km_df$group <- factor(sub("group=", "", as.character(km_sum$strata)),
                      levels = c("Low", "Medium", "High"))
km_df <- km_df[km_df$time <= 10, ]

p4 <- ggplot(km_df, aes(time, inc, color = group, fill = group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_step(linewidth = 0.9) +
  scale_color_manual(values = c(Low = col_blue, Medium = "#D4A24C", High = col_red),
                     labels = c("Low score", "Medium score", "High score")) +
  scale_fill_manual(values = c(Low = col_blue, Medium = "#D4A24C", High = col_red),
                    guide = "none") +
  scale_y_continuous(labels = percent_format()) +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 0.12)) +
  labs(subtitle = "All-cause mortality by protein-score tertile (discovery)",
       x = "Follow-up time (years)", y = "Cumulative mortality (%)", color = NULL) +
  theme_nm(8) + theme(legend.position = c(0.02, 0.98),
                      legend.justification = c(0, 1))

# Panel 5: validation correlation
val <- read.csv(file.path(DATA_DIR, "score_validation"),
                stringsAsFactors = FALSE)
val$transition_score_z <- as.numeric(scale(val$transition_score))
rho5 <- cor(val$protein_score, val$transition_score_z, method = "spearman")
val_p <- val[val$transition_score_z <= quantile(val$transition_score_z, 0.99,
                                                na.rm = TRUE), ]

p5 <- ggplot(val_p, aes(protein_score, transition_score_z)) +
  geom_point(alpha = 0.40, size = 1.9, shape = 21, fill = col_blue, stroke = 0.4) +
  geom_smooth(method = "lm", color = col_red, fill = alpha(col_red, 0.18),
              linewidth = 0.8, se = TRUE) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.6,
           label = sprintf("Spearman rho = %.3f", rho5), size = 2.8) +
  labs(subtitle = "Protein score vs transition score (validation)",
       x = "Protein score", y = "Transition score (z)") +
  theme_nm(8)

# Panel 6: validation mortality comparison
pheno <- readxl::read_excel(file.path(DATA_DIR, "validation"))
coef_df <- read.csv(file.path(DATA_DIR, "selected_proteins"),
                    stringsAsFactors = FALSE)
colnames(coef_df) <- c("protein", "gene", "coefficient")
matched <- intersect(coef_df$protein, colnames(pheno))
coef_df <- coef_df %>% filter(protein %in% matched) %>%
  arrange(match(protein, matched))
pheno$protein_score <- as.numeric(scale(as.matrix(pheno[, matched])) %*%
                                    coef_df$coefficient)
pheno$mortality <- ifelse(is.na(pheno$mortality), 0, pheno$mortality)
w <- wilcox.test(protein_score ~ mortality, data = pheno, exact = FALSE)$p.value

p6 <- ggplot(pheno, aes(factor(mortality), protein_score, fill = factor(mortality))) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.35) +
  geom_jitter(aes(color = factor(mortality)), width = 0.14,
              alpha = 0.30, size = 1.0) +
  scale_fill_manual(values = c(`0` = col_blue, `1` = col_red), guide = "none") +
  scale_color_manual(values = c(`0` = col_blue, `1` = col_red), guide = "none") +
  scale_x_discrete(labels = c("Alive", "Deceased")) +
  annotate("text", x = 1.5, y = max(pheno$protein_score, na.rm = TRUE),
           label = sprintf("P = %.4f", w), size = 2.8, fontface = "bold") +
  labs(subtitle = "Validation cohort: mortality comparison",
       x = NULL, y = "Protein score") +
  theme_nm(8)

ggsave(file.path(out_dir, "main.pdf"),
       p1 + p2 + p3 + p4 + p5 + p6 +
         plot_layout(design = "ABC\nDEF", widths = c(1, 1, 1), heights = c(1, 1)),
       width = 210, height = 150, units = "mm")


# 12. pathway enrichment
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(clusterProfiler); library(org.Hs.eg.db)
  library(DOSE); library(enrichplot)
})

set.seed(42)
out_dir <- file.path(OUT_DIR, "enrichment")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sel_enr <- read.csv(file.path(DATA_DIR, "selected_proteins"),
                    stringsAsFactors = FALSE)
sel_enr$protein_lower <- tolower(sel_enr$protein)
map <- read.delim(file.path(DATA_DIR, "mapping"),
                  stringsAsFactors = FALSE)[, 1:2]
names(map) <- c("from", "to"); map$from <- tolower(map$from)
sel_enr <- merge(sel_enr, map, by.x = "protein_lower", by.y = "from", all.x = TRUE)
gene_list <- unique(sel_enr$to[!is.na(sel_enr$to)])

gene_entrez <- bitr(gene_list, fromType = "SYMBOL", toType = "ENTREZID",
                    OrgDb = org.Hs.eg.db)
entrez_ids <- gene_entrez$ENTREZID

go_bp <- enrichGO(entrez_ids, OrgDb = org.Hs.eg.db, ont = "BP",
                  pvalueCutoff = 0.05, pAdjustMethod = "BH", readable = TRUE)
go_cc <- enrichGO(entrez_ids, OrgDb = org.Hs.eg.db, ont = "CC",
                  pvalueCutoff = 0.05, pAdjustMethod = "BH", readable = TRUE)
go_mf <- enrichGO(entrez_ids, OrgDb = org.Hs.eg.db, ont = "MF",
                  pvalueCutoff = 0.05, pAdjustMethod = "BH", readable = TRUE)
kegg  <- enrichKEGG(entrez_ids, organism = "hsa",
                    pvalueCutoff = 0.05, pAdjustMethod = "BH")
do    <- enrichDO(entrez_ids, ont = "DO", pvalueCutoff = 0.05,
                  pAdjustMethod = "BH", readable = TRUE)

enrichment_objects <- list(GO_BP = go_bp, GO_CC = go_cc, GO_MF = go_mf,
                           KEGG = kegg, DO = do)
for (nm in names(enrichment_objects)) {
  df <- as.data.frame(enrichment_objects[[nm]])
  if (nrow(df) > 0) {
    write.csv(df, file.path(out_dir, nm), row.names = FALSE)
  }
}


# 13. redundancy reduction and functional modules
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({ library(simplifyEnrichment); library(stringr) })

set.seed(123)
wang_cutoff <- 0.7; jaccard_cutoff1 <- 0.7; jaccard_cutoff2 <- 0.1; top_n <- 10

go_bp <- read.csv(file.path(out_dir, "GO_BP"), stringsAsFactors = FALSE)
go_cc <- read.csv(file.path(out_dir, "GO_CC"), stringsAsFactors = FALSE)
go_mf <- read.csv(file.path(out_dir, "GO_MF"), stringsAsFactors = FALSE)
kegg  <- read.csv(file.path(out_dir, "KEGG"),  stringsAsFactors = FALSE)

reduce_go <- function(df, ont, cutoff = wang_cutoff) {
  if (nrow(df) <= 1) return(df)
  sim <- GO_similarity(df$ID, ont = ont, db = "org.Hs.eg.db", measure = "Wang")
  sim[is.na(sim)] <- 0; diag(sim) <- 0
  adj <- sim; adj[adj < cutoff] <- 0
  g <- graph_from_adjacency_matrix(adj, mode = "undirected",
                                   weighted = TRUE, diag = FALSE)
  comm <- cluster_edge_betweenness(g, weights = E(g)$weight)
  df$module <- membership(comm)[df$ID]
  df %>% group_by(module) %>% slice_min(p.adjust, n = 1, with_ties = FALSE) %>%
    ungroup() %>% select(-module)
}

bp_reps <- reduce_go(go_bp, "BP") %>% mutate(source = "GO-BP")
cc_reps <- reduce_go(go_cc, "CC") %>% mutate(source = "GO-CC")
mf_reps <- reduce_go(go_mf, "MF") %>% mutate(source = "GO-MF")
kegg    <- kegg %>% mutate(source = "KEGG")

kegg$geneID <- sapply(kegg$geneID, function(genes) {
  ids <- str_split(genes, "/")[[1]]
  sym <- suppressMessages(mapIds(org.Hs.eg.db, keys = ids, column = "SYMBOL",
                                 keytype = "ENTREZID", multiVals = "first"))
  paste(sym[!is.na(sym)], collapse = "/")
})

common_cols <- c("ID", "Description", "p.adjust", "geneID", "source")
merged <- bind_rows(bp_reps[, common_cols], cc_reps[, common_cols],
                    mf_reps[, common_cols], kegg[, common_cols])

jaccard_sim_matrix <- function(gene_id_vec, ids) {
  genes <- str_split(gene_id_vec, "/"); n <- length(genes)
  m <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) if (i != j) {
    m[i, j] <- length(intersect(genes[[i]], genes[[j]])) /
      length(union(genes[[i]], genes[[j]]))
  }
  rownames(m) <- colnames(m) <- ids; m
}

adj1 <- jaccard_sim_matrix(merged$geneID, merged$ID)
adj1[adj1 < jaccard_cutoff1] <- 0
g1 <- graph_from_adjacency_matrix(adj1, mode = "undirected",
                                  weighted = TRUE, diag = FALSE)
comm1 <- cluster_edge_betweenness(g1, weights = E(g1)$weight)
merged$module2 <- membership(comm1)[merged$ID]

final_reps <- merged %>% group_by(module2) %>%
  slice_min(p.adjust, n = 1, with_ties = FALSE) %>% ungroup() %>% select(-module2)
top10 <- final_reps %>% arrange(p.adjust) %>% slice_head(n = top_n)

adj2 <- jaccard_sim_matrix(top10$geneID, top10$ID)
adj2[adj2 < jaccard_cutoff2] <- 0
g2 <- graph_from_adjacency_matrix(adj2, mode = "undirected",
                                  weighted = TRUE, diag = FALSE)
comm2 <- cluster_edge_betweenness(g2, weights = E(g2)$weight)
top10$functional_module <- as.factor(membership(comm2)[top10$ID])
top10$neglog10p <- -log10(top10$p.adjust)

write.csv(final_reps, file.path(out_dir, "final_pathways.csv"),
          row.names = FALSE)
write.csv(top10, file.path(out_dir, "top_pathways.csv"),
          row.names = FALSE)

# Pathway similarity network
g_nw <- graph_from_adjacency_matrix(jaccard_sim_matrix(top10$geneID, top10$ID),
                                    mode = "undirected", weighted = TRUE,
                                    diag = FALSE)
nw_graph <- tbl_graph(
  nodes = top10 %>% mutate(node = ID, label = Description,
                           functional_module = factor(functional_module)) %>%
    select(node, label, neglog10p, functional_module),
  edges = igraph::as_data_frame(g_nw, what = "edges") %>%
    rename(weight_jaccard = weight),
  directed = FALSE
)

p7 <- ggraph(nw_graph, layout = "fr") +
  geom_mark_hull(aes(x, y, fill = functional_module, group = functional_module),
                 concavity = 3, expand = unit(4, "mm"), alpha = 0.18, color = NA) +
  geom_edge_link(aes(width = weight_jaccard), color = "#B7B2AC", alpha = 0.65) +
  geom_node_point(aes(size = neglog10p, fill = functional_module),
                  shape = 21, color = "#6E6A66", stroke = 0.4) +
  geom_node_text(aes(label = label), repel = TRUE, size = 2.5) +
  scale_fill_manual(values = c(`1` = "#C2948E", `2` = "#7E9CAE",
                               `3` = "#8BA58B", `4` = "#B39E8F"),
                    name = "Functional module") +
  theme_void() + labs(subtitle = "Pathway similarity network")

top10 <- top10 %>%
  mutate(Count = sapply(strsplit(geneID, "/"), length)) %>%
  arrange(desc(Count), p.adjust)

p8 <- ggplot(top10, aes(Count, reorder(Description, Count))) +
  geom_point(aes(size = neglog10p, fill = p.adjust), shape = 21, color = "black") +
  scale_fill_gradientn(colours = c("#A8524A", "#C98D83", "#E4CFC6",
                                   "#D6DCE0", "#AFC7D9", "#7FA8C9"),
                       trans = "log10", name = "p.adjust") +
  labs(subtitle = "Pathway enrichment", x = "No. of genes (Count)", y = NULL) +
  theme_minimal(base_size = 7.5)

ggsave(file.path(out_dir, "network.pdf"), p7,
       width = 9, height = 6)
ggsave(file.path(out_dir, "dotplot.pdf"),
       p7 | p8, width = 210, height = 110, units = "mm")
