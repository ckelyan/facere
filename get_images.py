import requests as req
import pandas as pd

r = req.get('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt')

print(r.text)