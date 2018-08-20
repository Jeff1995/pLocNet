#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(dplyr)
    library(rhdf5)
})

message("Reading data...")
ppi <- read.table("../data/raw/9606.protein.actions.v10.5.txt.gz",
                  header = TRUE, sep = "\t")
ppi_clean <- ppi %>%
    filter(mode == "binding") %>%  # Physical binding only
    select(item_id_a, item_id_b, score)
stopifnot(length(setdiff(
    unique(ppi_clean$item_id_a),
    unique(ppi_clean$item_id_b)
)) == 0)

protein_id <- as.character(unique(ppi_clean$item_id_a))
protein_id_lut <- 1:length(protein_id)
names(protein_id_lut) <- protein_id

message("Filling matrix...")
ppi_mat <- matrix(
    0, nrow = length(protein_id), ncol = length(protein_id)
)
for (i in 1:nrow(ppi_clean)) {
    ppi_mat[
        protein_id_lut[as.character(ppi_clean$item_id_a[i])],
        protein_id_lut[as.character(ppi_clean$item_id_b[i])]
    ] <- ppi_clean$score[i]
}
ppi_mat_bool <- ppi_mat > 0
stopifnot(all(ppi_mat_bool == t(ppi_mat_bool)))
# all(ppi_mat == t(ppi_mat)) == FALSE

message("Saving results...")
output_file <- "../data/preprocessed/ppi.h5"
if (file.exists(output_file)) {
    stopifnot(file.remove(output_file))
}
h5write(protein_id, output_file, "protein_id")
h5write(ppi_mat, output_file, "mat")
h5write(ppi_mat_bool, output_file, "mat_bool")
