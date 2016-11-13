import sys
import os

from whoosh.fields import Schema, TEXT, STORED
from whoosh.index import create_in, open_dir
from whoosh.query import *
from whoosh.qparser import QueryParser

#creating the schema
schema = Schema(tax_id=STORED,
                name=TEXT(stored=True))

#creating the index
if not os.path.exists("index_test"):
    os.mkdir("index_test")

ix = create_in("index_test",schema)
# ix = open_dir("index")
writer = ix.writer()
writer.add_document(tax_id="17",name=u"Methyliphilus methylitrophus")
writer.add_document(tax_id="17",name=u"Methylophilus methylotrophus Jenkins et al. 1987")
writer.add_document(tax_id="45",name=u"Chondromyces lichenicolus")
writer.commit()


with ix.searcher() as searcher:
    # myquery = And([Term("name",u"Chondromyces")])
    myquery = QueryParser("name", ix.schema).parse(u'Chondromyces')
    print myquery
    results = searcher.search(myquery)
    for result in results:
        print result

# with ix.searcher() as searcher:
#     query = QueryParser("name", ix.schema).parse(u'Chondromyces')
#     results = searcher.search(query)
#     for result in results:
#         print result