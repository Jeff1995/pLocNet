#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(rhdf5)
    library(pROC)
    library(mccr)
    library(reshape2)
    library(ggplot2)
})


#===============================================================================
#
#  Preparation
#
#===============================================================================
root <- "../result"
features <- c("3mer", "cnnvae", "cnn")
mlp_models <- paste(features, "mlp", sep = "_")
gcn_models <- paste(features, "gcn", sep = "_")
models <- c("cnn", mlp_models, "gn", gcn_models)
locs <- read.table("../data/preprocessed/used_labels.txt",
                   sep = "\t", header = FALSE, stringsAsFactors = FALSE)[, 1]
folds <- 0:4

evaluate_performance <- function(true, pred, cutoff = NULL) {
    stopifnot(all(dim(true) == dim(pred)))
    roc <- lapply(
        1:ncol(true),
        function(i) roc(true[, i], pred[, i], direction = "<")
    )
    auc <- sapply(roc, `[[`, "auc")
    if (is.null(cutoff)) {
        cutoff <- sapply(roc, function(roc) {
            cutoff <- coords(
                roc, x = "best", ret = "threshold", best.weights = c(
                    1, max(sum(roc$response) / length(roc$response), 0.25)
                )  # cutoff may become inf if prevalence is too skewed
            )
            single_cutoff <- cutoff[ceiling(length(cutoff) / 2)]
            if (!is.finite(single_cutoff)) browser()
            single_cutoff
        })
    }
    pred <- t(t(pred) > cutoff)
    acc <- colSums(true == pred) / nrow(true)
    precision <- colSums(true & pred) / colSums(pred)
    recall <- colSums(true & pred) / colSums(true)
    mcc <- sapply(1:ncol(true), function(i) mccr(true[, i], pred[, i]))
    list(auc = auc, cutoff = cutoff, acc = acc,
         precision = precision, recall = recall, mcc = mcc)
}

row_num <- length(models) * length(folds) * length(locs)
empty_performance_df <- data.frame(
    model = factor(character(row_num), levels = models),
    loc = factor(character(row_num), levels = locs),
    fold = integer(row_num),
    AUC = numeric(row_num),
    Accuracy = numeric(row_num),
    Precision = numeric(row_num),
    Recall = numeric(row_num),
    MCC = numeric(row_num),
    cutoff = numeric(row_num),
    stringsAsFactors = FALSE
)


#===============================================================================
#
#  Compute performance
#
#===============================================================================
compute_performance <- function(performance_df, mode, locs. = locs) {
    row_idx <- 1
    for (model in models) {
        for (fold in folds) {
            cat(sprintf("[%3d/%3d]\r", row_idx, nrow(performance_df)))
            file <- sprintf("%s/%s/result.h5", root, paste(
                c(model, sprintf("fold%d", fold)), collapse = "_"
            ))
            if (mode == "train") {
                performance <- evaluate_performance(
                    t(h5read(file, "train/true")),
                    t(h5read(file, "train/pred"))
                )
            } else {  # mode == "test"
                performance <- evaluate_performance(
                    t(h5read(file, "test/true")),
                    t(h5read(file, "test/pred")),
                    cutoff = performance_df[
                        row_idx:(row_idx + length(locs.) - 1),
                        "cutoff"
                    ]  # "cutoff" column should be available
                )
            }
            stopifnot(length(performance$auc) == length(locs.) &&
                      length(performance$acc) == length(locs.))
            block_idx <- row_idx:(row_idx + length(locs.) - 1)
            performance_df[block_idx, "model"] <- model
            performance_df[block_idx, "loc"] <- locs
            performance_df[block_idx, "fold"] <- fold
            performance_df[block_idx, "AUC"] <- performance$auc
            performance_df[block_idx, "Accuracy"] <- performance$acc
            performance_df[block_idx, "Precision"] <- performance$precision
            performance_df[block_idx, "Recall"] <- performance$recall
            performance_df[block_idx, "MCC"] <- performance$mcc
            performance_df[block_idx, "cutoff"] <- performance$cutoff
            row_idx <- row_idx + length(locs.)
        }
    }
    performance_df
}

message("Computing training set performance...")
train_performance_df <- compute_performance(empty_performance_df, mode = "train")
message("Computing testing set performance...")
test_performance_df <- compute_performance(train_performance_df, mode = "test")


#===============================================================================
#
#  Plotting
#
#===============================================================================
gp_theme <- theme(
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank()
)

plot_performance <- function(performance_df, path, prefix = "") {
    for (variable in c("AUC", "Accuracy", "Precision", "Recall", "MCC")) {
        cat(sprintf("%9s\r", variable))
        df <- melt(performance_df, id.vars = c("model", "loc"), measure.var = variable)
        gp <- ggplot(df, aes(x = model, y = value, col = model, fill = model)) +
            geom_boxplot(alpha = 0.5) + facet_wrap(~loc) +
            scale_y_continuous(name = variable) + gp_theme
        ggsave(sprintf("%s/%s_performance_%s.pdf", path, prefix, variable),
               gp, width = 10, height = 8)
    }
}

message("Plotting training set performance...")
plot_performance(train_performance_df, path = root, prefix = "train")
message("Plotting testing set performance...")
plot_performance(test_performance_df, path = root, prefix = "test")

message("Done!     ")
