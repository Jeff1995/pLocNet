# data

## raw

* `9606.protein.links.v10.5.txt.gz`: Human protein interactions from STRING database
* `9606.protein.actions.v10.5.txt.gz`: Human protein interactions from STRING database (**preferred**)
    - Interaction types are annotated, so that we can extract only physical interactions
* `BIOGRID-ORGANISM-Homo_sapiens-3.4.163.tab2.txt.gz`: Human gene interactions from BIOGRID database
* `LOCATE_human_v6_20081121.xml.gz`: Human protein subcellular localizations from LOCATE database
* `LOCATE_v6.xsd.gz`: LOCATE database XML schema
* `uniprot_sprot.xml.gz`: Complete UniProtKB database (**preferred**)
    - Contains all kinds of protein IDs
    - Contains subcellular location annotations
    - Contains protein sequence
* `uniprot.xsd.gz`: UniProtKB database XML schema
* `locations-all.tab.gz`: UniProtKB subcellular location controlled vocabulary

## preprocessed

* `localization.tsv.gz`: Subcellular locations extracted from `uniprot_sprot.xml.gz`
* `ppi.h5`: Protein-protein interaction network extracted from `9606.protein.actions.v10.5.txt.gz`
    - `/mat` is the interaction matrix whose elements are interaction scores from `9606.protein.actions.v10.5.txt.gz`.
      **Note that this matrix is not symmetric (scores of several pairs are not symmetric).**
    - `/mat_bool` is the binarized version of `/mat`
      **This matrix is symmetric**
    - `/protein_id` is a list of STRING protein IDs, which serves as the row/column name of `/mat` and `/mat_bool`
* `sequence.fasta.gz`: Protein sequences extracted from `uniprot_sprot.xml.gz`
    - `sequenceNN.fasta.gz`: Cluster representatives after clustering `sequence.fasta.gz` with `cd-hit`
    - `sequenceNN.clstr.gz`: Cluster members after clustering `sequence.fasta.gz` with `cd-hit`
    - `NN` is the `cd-hit` similarity threshold, e.g. NN=90 means that sequences with \>90% sequence similarity
      are clustered together, i.e. remaining cluster representatives have maximal similarity \<90%
