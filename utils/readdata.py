from sas7bdat import SAS7BDAT

dirt = 



with SAS7BDAT('foo.sas7bdat') as f:
    for row in f:
        print row
