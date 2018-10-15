"""Data preprocessing: 
    - Read Wikipedia Dump (.bz) document-by-document;
    - Spacy parse;
    - Ollie parse;
    - Save to a file of event list, where documents are separated by `\n`. 

Author: Su Wang
"""

import bz2
from collections import Counter
import os
import spacy
import subprocess

PARSER = spacy.load("en")


def get_root(phrase):
    """Given a subject/predicate/dobject phrase, return the head if there is one."""
    parsed = PARSER(phrase)
    for token in parsed:
        if token.dep_ == "ROOT":
            if token.lemma_ == "be":
                head = token.head.lemma_
            else:
                head = token.lemma_
            if head in ["be", "get", "do", "have"]:
                return False
            return head
    return False


def parse_triple(triple):
    """Clean up the phrases in a triple."""
    triple_string = triple.split("; ")
    # Spacy parse to get the head of phrase.
    subject = get_root(triple_string[0][1:]) # delete init `(`.
    predicate = get_root(triple_string[1])
    dobject = get_root(triple_string[2][:-1]) # delete end `)`.
    if subject and predicate and dobject:
        return subject, predicate, dobject
    return False

    
def ollie_file_to_clean_triples(ollie_file_path, out_file_path, write_mode="a"):
    """Clean up an Ollie-extracted event triple and write/append to an aggregation file."""
    with open(ollie_file_path) as ollie_file, \
         open(out_file_path, write_mode) as out_file:
        for line in ollie_file:
            line = line.strip()
            if line.startswith("0"):
                try:
                    _, triple = line.split(": ")
                    parsed_triple = parse_triple(triple)
                    if parsed_triple:
                        subject, predicate, dobject = parsed_triple
                        out_file.write(subject + "\t" + predicate + "\t" + dobject + "\n")
                except:
                    continue # skip occassional bad cases.
            else:
                continue
        out_file.write("\n") # separate document with "\n".
                

def extract_events_from_wiki_bz2(wiki_bz2_file_path, # one wiki.bz2 with multi-documents.
                                 document_file_path, # temp .txt for one document.
                                 ollie_file_path,    # temp .txt for one document.
                                 target_file_path,   # global .txt for all documents.
                                 write_mode="a",     # "w" or "a".
                                 print_every=1000): 
    """Run preprocessing pipeline."""
    document_count = 0
    print("Processing document %s ...\n" % os.path.basename(wiki_bz2_file_path))
    for line in bz2.BZ2File(wiki_bz2_file_path): # one line => one document.
        document_count += 1
        sentences = line.strip().decode("utf-8").split(". ") # sentence segmentation.
        # Write the sentences of 1 document to the document temp file.
        with open(document_file_path, "w") as document_file:
            for sentence in sentences:
                sentence = sentence[1:] if sentence.startswith(" ") else sentence
                document_file.write(sentence + "\n") 
        # Ollie-parse to extract events and output to ollie file tmep file.
        subprocess.call(["java", "-Xmx512m", "-jar", "ollie-app-latest.jar", 
                         document_file_path, "-o", ollie_file_path])
        ollie_file_to_clean_triples(ollie_file_path, target_file_path, write_mode)
        if document_count % print_every == 0:
            print("  ... processed %d documents." % document_count)
    print("\nDone!\n")
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_bz2_file_path", type=str)
    parser.add_argument("--document_file_path", type=str)
    parser.add_argument("--ollie_file_path", type=str)
    parser.add_argument("--target_file_path", type=str)
    parser.add_argument("--print_every", type=int)
    args = parser.parse_args()
    
    extract_events_from_wiki_bz2(wiki_bz2_file_path=args.wiki_bz2_file_path, 
                                 document_file_path=args.document_file_path, 
                                 ollie_file_path=args.ollie_file_path,   
                                 target_file_path=args.target_file_path,
                                 write_mode="a",
                                 print_every=args.print_every)   