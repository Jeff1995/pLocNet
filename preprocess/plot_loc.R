#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
})

loc_df <- read.table("../data/preprocessed/localization.tsv.gz", sep = "\t")
colnames(loc_df) <- c("protein_id", "localization", "evidence")

loc_df_clean <- loc_df %>%
    filter(evidence != "None") %>%
    select(protein_id, localization) %>%
    distinct()
loc_occurence <- as.data.frame(table(loc_df_clean$localization))
colnames(loc_occurence) <- c("localization", "count")
loc_occurence <- loc_occurence[order(loc_occurence$count, decreasing = TRUE), ]

gp <- ggplot(loc_occurence, aes(x = localization, y = count)) +
        geom_bar(stat = "identity") +
            scale_x_discrete(limits = loc_occurence$localization) +
                coord_flip()
ggsave("../data/summary/loc.pdf", gp, height = 30)

gp <- ggplot(loc_occurence[1:20, ], aes(x = localization, y = count)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(limits = loc_occurence$localization[1:20]) +
    coord_flip() + theme(
        axis.text.y = element_text(size = 10),
        axis.text.x = element_text(size = 12)
    )
ggsave("../data/summary/loc_short.pdf", gp)
