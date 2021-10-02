# cleaning.py
"""
Contains functions to parse and clean texts from .pdf files
"""
import re
import fitz
import copy
import warnings
from math import log10, floor

def parse_file(filepath):
    """Parses pdf file, returns dictionary where key=pagenum, value=list of
    text/image block strings (paragraphs) for pdf.

    Args:
        filepath (str):  Filepath for .pdf file

    Returns:
        dict: Dictionary of paragraphs where key=pagenum and value=paragraph
        list.
    """
    with fitz.open(filepath) as doc:
        block_dict = {(idx + 1): page.getText("blocks") for idx, page in enumerate(doc)}
        block_dict = {
            key: [block[4] for block in value] for key, value in block_dict.items()
        }
    return block_dict

def round_sig(x, sig=2):
    """
    Returns number rounded to 2 significant figures
    """
    return round(x, sig-int(floor(log10(abs(x))))-1)


def search_and_count_chars(dictionary, lookup):
    """
    Returns the count of characters in all dictionary values
    that contain the specified lookup regex string
    """

    c = (sum(sum(len(re.findall(lookup,s)) for s in subList) for subList in dictionary.values()))
    return c

def rep_char_search(pdf_dict):
    """
    Checks a parsed and cleaned pdf dictionary for remaining
    replacement characters. Does not modify the input.   
    """

    replacement_characters = search_and_count_chars(pdf_dict, '�')
    all_characters = search_and_count_chars(pdf_dict, r'[^ ]')
    percent_non_readable = (100*replacement_characters/all_characters)
    
    if replacement_characters != 0:
        warnings.warn("Warning! This document contains {0} non-readable characters, which is {1}% of the entire document. These characters will now be removed.".format(replacement_characters,round_sig(percent_non_readable,2)), stacklevel=2)
              
    return pdf_dict

def find_tables(pdf_dict):
    """Finds tables within a parsed pdf file.

    Args: 
        pdf_dict: Dictionary of paragraphs where key=pagenum and
        value=paragraph list
    
    Returns:
        table_dict: Dictionary of tables where key=tablenum and
        value=table_position_dict where table_position_dict is a dictionary
        where key=[page, paragraph, text] and
        value=[page_num, paragraph_num, raw_table_text]
        
    Lucas das Dores 03/09/2021
    """
    
    table_dict = {}
    
    # matches numerical tables with no whitespace between entries
    table_matcher1= re.compile('\S\n[\d\W]')
    
    # matches tables with deliberate whitespaces between entries
    table_matcher2= re.compile('\s\n[\d\s]') 
    
    
    i= 0
    for page_num, paragraphs in pdf_dict.copy().items():
        for paragraph_num, text in enumerate(paragraphs):
            
            # This if statement decides what should be interpreted
            # as a "table string" on the text.
            # Right now, it is set to identify as a table a string that
            # has more than 4 newline characters surrounded by non white space
            # characters or a string with at least three
            # newline spaces deliberately surrounded by white spaces
            # This 'sensitivity of tables' can be modified according
            # the need and aspect of documents parsed.
            
            if (len(table_matcher1.findall(text))>=4 or len(table_matcher2.findall(text))>=3):
                i+=1
                table_position_dict = {'page':page_num,
                                       'paragraph': paragraph_num+1,
                                       'raw_table_text':text}
                table_dict[i] = table_position_dict
    return table_dict

def join_multiline_tables(table_dict):
    """Joins tables that are expressed in several paragraphs on a document.
    
    This function tries to address the issue of finding
    multiple table strings in sequence, which in general
    means it is a single table spread accross several consecutive
    paragraphs outputted by the parse_file function.
    
    Args:
        table_dict
        
    Returns:
        joined_table_dict
        
    Lucas das Dores 03/09/2021
    """
    
    # The idea is to loop over all the items marked as a table string
    # and check whether they are in the same page and whether the previous
    # paragraph outputted by the parser is also a table string.
    # If both conditions are true, then join them.
    
    joined_table_dict ={}
    i=1
    
    # 
    for k,v in table_dict.items():
        
        # The first part of the first joined table
        # is just the first table on the dictionary
        if k ==1:
            joined_table_dict[i] = table_dict[k]
        else:
            # Check if the previous table is on the same page
            # and if the paragraph is the previous paragraph
            
            previous_table_in_same_page = (table_dict[k-1]['page']==table_dict[k]['page'])
            previous_table_is_previous_paragraph = (table_dict[k]['paragraph']==table_dict[k-1]['paragraph']+1)
            
            # If both conditions are true and
            # the string is not just empty space add a newline character
            # and the text of the current table
            
            if previous_table_in_same_page and previous_table_is_previous_paragraph:
                if not table_dict[k]['raw_table_text'].isspace():
                    joined_table_dict[i]['raw_table_text'] += ('\n' + table_dict[k]['raw_table_text'])
            else:
                i+=1
                joined_table_dict[i] = table_dict[k]
    return(joined_table_dict)

def clean_table_text(raw_table_text):
    """
    Removes numerical data from a table string and returns only
    text data contained on it. The idea of the function is to
    give only textual context for our matching
    functions on the preprocessing.py script.
    
    Args:
        raw_table_text (str)
    Returns:
        clean_table_text (str)
        
    Lucas das Dores 03/09/2021
    """
    # First we split the tables at all newline characters
    table_entries = raw_table_text.split('\n')
    
    # Define matcher for alphabetical entries
    word_match = re.compile('[a-zA-Z]+')
    
    # For all table entries remove those which are not alphabetical
    for entry in table_entries[:]:
        if word_match.match(entry) == None:
            table_entries.remove(entry)
    cleaned_table_text = ' '.join(table_entries).strip()
    
    return (cleaned_table_text)

def toy_clean(pdf_dict, table_contents=False):
    """ Cleans text outputted by parse_file function. More specifically:
            Converts extra whitespace, newline and non-breaking characters
            into single space, 
            Removes all image strings,
            Removes Adobe InDesign strings.
		    Removes non-readable characters
            Includes summary of table words in place of tables (optional)
        
        Args: 
           pdf_dict (dict): Dictionary of paragraphs where
               key=pagenum and value=paragraph list
           table_contents (bool): Boolean to decide wether to include
                       the text identified to be inside tables or not.
                       DOES NOT INCLUDE NUMERICAL DATA OF TABLES.
        
        Returns:
            clean_pdf_dict (dict): Dictionary
            of paragraphs where key=pagenum and value=paragraph list
    """
    rep_char_search(pdf_dict)
    # Creates deepcopy of the clean dictionary to avoid
    # unexpected behaviour from altering it
    clean_pdf_dict = copy.deepcopy(pdf_dict)
    
    # Identify tables and store them
    raw_tables = find_tables(clean_pdf_dict)
    # Join all tables that are possible spread in sevral paragraphs
    joined_tables = join_multiline_tables(raw_tables)
    
    # Notice that when we used join_multiline_tables
    # we could have several paragraphs of tables but have only retained
    # the information on the paragraph of the first table to be joined.
    # 
    # The following lines do the following:
    # we will lopp through all the raw tables (not joined) identified and
    # use the index i to loop over all joined tables.
    #
    # If the raw table page and paragraph coincides with that of a joined table
    # replace the table by the joined table.
    # if this is not the case replace it with an empty string.
    
    i=1
    for table in raw_tables.values():
        
        # If i is bigger than the length of the joined tables
        # we have finished looping through the joined tables before
        # finishing looping through the raw tables
        # so replace the rest of the tables with an empty string
        
        if i >= len(joined_tables):
            clean_pdf_dict[table['page']][table['paragraph']-1] = ''
            
        # If table the raw table does not coincide with
        # the beginning of joined table replace it by an empty string
        
        elif table['page'] != joined_tables[i]['page'] or table['paragraph'] != joined_tables[i]['paragraph']:
            clean_pdf_dict[table['page']][table['paragraph']-1] = ''
            
        # else the raw table page and paragraph coincides with that
        # of a joined table and we replace the table by the joined table.
        
        else:
            clean_text = clean_table_text(joined_tables[i]['raw_table_text'])
            clean_pdf_dict[table['page']][table['paragraph']-1] = f'<TABLE {i} CONTENTS({clean_text}) ENDTABLE>' 
            i+=1
            
    # If we do not want table contents do not run the code below    
        
    if not table_contents:
        clean_pdf_dict.update((k, [x for x in v if '<TABLE' != x[:6]]) for k,v in clean_pdf_dict.items())
        
    # Updates the text dictionary as described on docstring    
    clean_pdf_dict.update((k, [x for x in v if not '<image:' in x]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [x for x in v if not '.indd' in x]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [x for x in v if not '�' in x]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [x for x in v if not u'\uf0b7' in x]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [x for x in v if x !='']) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [x for x in v if not '–' in x]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [re.sub(r'\s+', ' ', u"%s"%(x)) for x in v]) for k,v in clean_pdf_dict.items())
    clean_pdf_dict.update((k, [re.sub(r'[^ -~]', ' ', u"%s"%(x)) for x in v]) for k,v in clean_pdf_dict.items())

    return clean_pdf_dict 
