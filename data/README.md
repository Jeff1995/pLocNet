# data

## raw

* `9606.protein.links.v10.5.txt.gz`: Human protein interactions from string database (**preferred**)
* `BIOGRID-ORGANISM-Homo_sapiens-3.4.163.tab2.txt.gz`: Human gene interactions from BIOGRID database
* `LOCATE_human_v6_20081121.xml.gz`: Human protein subcellular localizations from LOCATE database
* `LOCATE_v6.xsd.gz`: LOCATE database XML schema
* `uniprot_sprot.xml.gz`: Complete UniProtKB database (**preferred**)
    - Contains all kinds of protein ids
    - Contains subcellular location annotations
    - Contains protein sequence
* `uniprot.xsd.gz`: UniProtKB database XML schema
* `locations-all.tab.gz`: UniProtKB subcellular location controlled vocabulary

## preprocessed

* `localization.tsv.gz`: Subcellular locations extracted from `uniprot_sprot.xml.gz`
* `ppi.txt.gz`: Directly links to `9606.protein.links.v10.5.txt.gz`
* `sequence.fasta.gz`: Protein sequences extracted from `uniprot_sprot.xml.gz`
