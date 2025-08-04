import urllib.request

url = "https://cmpe.boun.edu.tr/~hakime.ozturk/source/embeddings/drug.pubchem.canon.l8.ws20.txt"
filename = "drug.pubchem.canon.l8.ws20.txt"

urllib.request.urlretrieve(url, filename)