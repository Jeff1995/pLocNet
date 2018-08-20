#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(Biostrings)
    library(ggplot2)
})

gp <- ggplot(data.frame(
    length = fasta.seqlengths("../data/preprocessed/sequence.fasta.gz")
), aes(x = length)) + geom_histogram(bins = 25) + scale_x_log10()
ggsave("../data/summary/seqlen.pdf", gp)
