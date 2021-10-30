# Created by Henry O'Scannlain-Miller
# This is a simple server that returns the first level of the Google Product Taxonomy
# that is predicted from a Torch model.
# Google Product Taxonomy: (https://www.google.com/basepages/producttype/taxonomy.en-US.txt)

from flask import Flask, request
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = torch.load('21categories_model.torch')
class_names = ['Animals & Pet Supplies',
 'Apparel & Accessories',
 'Arts & Entertainment',
 'Baby & Toddler',
 'Business & Industrial',
 'Cameras & Optics',
 'Electronics',
 'Food, Beverages & Tobacco',
 'Furniture',
 'Hardware',
 'Health & Beauty',
 'Home & Garden',
 'Luggage & Bags',
 'Mature',
 'Media',
 'Office Supplies',
 'Religious & Ceremonial',
 'Software',
 'Sporting Goods',
 'Toys & Games',
 'Vehicles & Parts']

app = Flask(__name__)

# handle request
# bad requests automaticaly return 400
@app.route('/class', methods=['GET'])
def classify():
    file = request.files['image']
    image = Image.open(file)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    _, output = torch.max(model(x), 1)
    predicted_category = class_names[output]
    return predicted_category

# start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)

