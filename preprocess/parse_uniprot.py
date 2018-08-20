#!/usr/bin/env python

import gzip
from lxml import etree
from tqdm import tqdm

NAMESPACE = "http://uniprot.org/uniprot"


def is_human(entry):
    organism = list(entry.iterchildren("{%s}organism" % NAMESPACE))
    assert len(organism) == 1
    for reference in filter(
        lambda x: x.get("type") == "NCBI Taxonomy",
        organism[0].iterchildren("{%s}dbReference" % NAMESPACE)
    ):
        if reference.get("id") == "9606":
            return True
    return False


def get_id(entry):
    results = []
    for item in filter(
        lambda x: x.get("type") == "STRING",
        entry.iterchildren("{%s}dbReference" % NAMESPACE)
    ):
        results.append(item.get("id"))
    assert len(results) <= 1
    return results[0] if results else None


def get_seq(entry):
    seq = list(entry.iterchildren("{%s}sequence" % NAMESPACE))
    assert len(seq) == 1
    return seq[0].text


def get_loc(entry):
    results = []
    for comment in filter(
        lambda x: x.get("type") == "subcellular location"
        and not list(x.iterchildren("{%s}molecule" % NAMESPACE)),
        entry.iterchildren("{%s}comment" % NAMESPACE)
    ):
        for subcellularLocation in comment.iterchildren(
            "{%s}subcellularLocation" % NAMESPACE
        ):
            for location in subcellularLocation.iterchildren(
                "{%s}location" % NAMESPACE
            ):
                results.append((location.text, location.get("evidence")))
    return results


def main():
    with gzip.open("../data/raw/uniprot_sprot.xml.gz", "rb") as uniprot_file, \
         gzip.open("../data/preprocessed/sequence.fasta.gz", "wt") as seq_file, \
         gzip.open("../data/preprocessed/localization.tsv.gz", "wt") as loc_file:
        print("Parsing XML file...")
        root = etree.parse(uniprot_file).getroot()
        print("Scanning entries...")
        entries = list(root.iterchildren("{%s}entry" % NAMESPACE))
        for entry in tqdm(entries, unit="entries"):
            if not is_human(entry):
                continue
            protein_id = get_id(entry)
            if not protein_id:
                continue
            seq_file.write(">%s\n" % protein_id)
            seq_file.write("%s\n" % get_seq(entry))
            for loc, evidence in get_loc(entry):
                loc_file.write("%s\t%s\t%s\n" % (protein_id, loc, evidence))


if __name__ == "__main__":
    main()
    print("Done!")
