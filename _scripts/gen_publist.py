#!/usr/bin/env python

"""
The script for generating the publication list
"""

import bibtexparser
import numpy as np
from bibtexparser import customization
from bibtexparser.bparser import BibTexParser

bibfile = '../../../../writing/CV/zhenwen.bib'
outfile = '../publications.md'

def getnames(names):
    """Make people names as firstnames surname. Should eventually combine up the two.

    :param names: a list of names
    :type names: list
    :returns: list -- Correctly formated names
    """
    tidynames = []
    for namestring in names:
        namestring = namestring.strip()
        if len(namestring) < 1:
            continue
        if ',' in namestring:
            namesplit = namestring.split(',', 1)
            last = namesplit[0].strip()
            firsts = [i.strip() for i in namesplit[1].split()]
        else:
            namesplit = namestring.split()
            last = namesplit.pop()
            firsts = [i.replace('.', '. ').strip() for i in namesplit]
        if last in ['jnr', 'jr', 'junior']:
            last = firsts.pop()
        for item in firsts:
            if item in ['ben', 'van', 'der', 'de', 'la', 'le']:
                last = firsts.pop() + ' ' + last
        tidynames.append(' '.join(firsts)+' '+last)
    return tidynames

def customize(record):
    record = customization.convert_to_unicode(record)
    record = customization.author(record)
    return record

def parse_bib_entry(entry):
    journaldetails = None
    otherinfo = ''
    if entry['type']=='inproceedings':
        pubname = entry['booktitle']
    elif entry['type']=='article':
        pubname = entry['journal']
        journaldetails = entry
    title = entry['title']
    authors = getnames(entry['author'])
    year = entry['year']
    if 'pages' in entry:
	    otherinfo += ', '+entry['pages']
    return authors, title, pubname, year, otherinfo

def produce_publication_entry(entry):
    if entry['type'] != 'article' and entry['type']!='inproceedings':
        return ''
    authors, title, pubname,year, otherinfo = parse_bib_entry(entry)
    
    authors = [name if name!='\\myfirstname \\mylastname' else '**Zhenwen Dai**' for name in authors ]
    entry_str = ''
    entry_str += '+   '+', '.join(authors[:-1])+' and '+authors[-1]+' ('+year+')  \n'
    entry_str += '    **'+title+'**  \n'
    entry_str += '    '+pubname+otherinfo+'  \n'
    entry_str += '\n'
    return entry_str

def make_head():
    return """---
layout: page
title: My publications
tags: [about, Jekyll, theme, responsive]
comments: false
share: false
---
"""

if __name__ == '__main__':
	# Load bibtex
	with open(bibfile, 'r') as f:
	    parser = BibTexParser()
	    parser.customization = customize
	    bib_db = bibtexparser.load(f, parser=parser)
	    f.close()

 	year_list = [int(e['year']) for e in bib_db.entries]
 	order_list = np.argsort(year_list)[::-1]

	#Produce publications.md
	with open(outfile,'w') as f:
	    f.write(make_head().encode('utf8'))
	    for i in order_list:
	        f.write(produce_publication_entry(bib_db.entries[i]).encode('utf8'))
	    f.close()    
