import pyux
import sciann
import json

sign = pyux.sign(sciann)

with open('api.json', 'w') as f:
    json.dump(sign, f)
