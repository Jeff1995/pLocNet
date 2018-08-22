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
* `localization.h5`: Subcellular locations encoded into a label matrix
    - `/mat`: Binary matrix of protein subcellular locations
    - `/protein_id`: Serves as row names of `/mat`
    - `/label`: Serves as column names of `/mat`
* `used_labels.txt`: Hand-picked subcellular localizations that are used for prediction
* `ppi.h5`: Protein-protein interaction network extracted from `9606.protein.actions.v10.5.txt.gz`
    - `/mat` is the interaction matrix whose elements are interaction scores from `9606.protein.actions.v10.5.txt.gz`.
      **Note that this matrix is not symmetric (scores of several pairs are not symmetric).**
    - `/mat_bool` is the binarized version of `/mat`.
      **This matrix is symmetric**
    - `/protein_id` is a list of STRING protein IDs, which serves as the row/column names of `/mat` and `/mat_bool`
* `sequence.fasta.gz`: Protein sequences extracted from `uniprot_sprot.xml.gz`
    - `sequenceNN.fasta.gz`: Cluster representatives after clustering `sequence.fasta.gz` with `cd-hit`
    - `sequenceNN.clstr.gz`: Cluster members after clustering `sequence.fasta.gz` with `cd-hit`
    - `NN` is the `cd-hit` similarity threshold, e.g. NN=90 means that sequences with \>90% sequence similarity
      are clustered together, i.e. remaining cluster representatives have maximal similarity \<90%
* `sequence.h5`: Protein sequences encoded into one-hot matrices
    - `/mat`: A 3D tensor, of which the first dimension is sequences, second is sequence length, last is amino acid identity
    - `/protein_id`: Serves as names of the first dimension of `/mat`
    - `/aa`: Serves as names of the last dimension of `/mat`
    - `sequenceNN.h5` files are encoded from `sequenceNN.fasta.gz` files correspondingly
