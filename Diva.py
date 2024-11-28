from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Connexion à MongoDB
client = MongoClient('mongodb+srv://aguess1874:Alex.1874@cluster0.mr9n5.mongodb.net/')
db = client['AirBnB']
collection = db['AirBnB']

@app.route('/')
def index():
    return render_template('index.html')

# Fonctionnalité de recherche
@app.route('/search', methods=['GET'])
def search():
    min_nights = request.args.get('min_nights')
    neighbourhood_group = request.args.get('neighbourhood_group')
    instant_bookable = request.args.get('instant_bookable')

    query = {'price': {'$exists': True, '$ne': ''}}
    if min_nights:
        query['minimum nights'] = {'$gte': int(min_nights)}
    if neighbourhood_group:
        query['neighbourhood group'] = neighbourhood_group
    if instant_bookable:
        query['instant_bookable'] = True

    results = list(collection.find(query, {'_id': 0, 'NAME': 1, 'price': 1, 'minimum nights': 1, 'instant_bookable': 1, 'neighbourhood group': 1}).sort('price', 1))
    return jsonify(results)

# Fonctionnalité de mise à jour
@app.route('/search_update', methods=['GET'])
def search_update():
    name = request.args.get('name')
    query = {'price': {'$exists': True, '$ne': ''}}
    if name:
        query['NAME'] = name

    result = collection.find_one(query, {'_id': 1, 'NAME': 1, 'price': 1, 'minimum nights': 1, 'instant_bookable': 1, 'neighbourhood group': 1})
    if result:
        return render_template('update.html', result=result)
    return jsonify({'error': 'Document non trouvé'}), 404

@app.route('/update', methods=['POST'])
def update():
    document_id = request.form.get('id')
    new_data = {}
    if request.form.get('price'):
        new_data['price'] = request.form.get('price')
    if request.form.get('min_nights'):
        new_data['minimum nights'] = int(request.form.get('min_nights'))
    if request.form.get('instant_bookable'):
        new_data['instant_bookable'] = request.form.get('instant_bookable') == 'on'
    if request.form.get('neighbourhood_group'):
        new_data['neighbourhood group'] = request.form.get('neighbourhood_group')

    result = collection.update_one({'_id': ObjectId(document_id)}, {'$set': new_data})
    if result.modified_count:
        return jsonify({'message': 'Mise à jour réussie'})
    return jsonify({'error': 'Échec de la mise à jour'}), 400

@app.route('/delete', methods=['POST'])
def delete():
    name = request.json.get('name')
    result = collection.delete_one({'NAME': name})
    if result.deleted_count:
        return jsonify({'message': 'Suppression réussie'})
    return jsonify({'error': 'Échec de la suppression'}), 400

# Fonctionnalité de graphiques
@app.route('/graphs')
def graphs():
    data = pd.DataFrame(list(collection.find({}, {'room type': 1, 'price': 1, 'minimum nights': 1, '_id': 0})))

    plt.figure(figsize=(10, 6))
    data['room type'].value_counts().plot(kind='bar')
    plt.title('Distribution des types de chambres')
    plt.savefig('static/room_type_bar.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    data['price'] = data['price'].apply(lambda x: float(x.replace('$', '').strip()))
    data['price'].plot(kind='hist', bins=50)
    plt.title('Distribution des prix')
    plt.savefig('static/price_hist.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    data['minimum nights'].plot(kind='box')
    plt.title('Distribution des nuits minimums')
    plt.savefig('static/min_nights_box.png')
    plt.close()

    return jsonify({'message': 'Graphiques créés'})

# Algorithmes de machine learning
@app.route('/ml')
def ml():
    data = pd.DataFrame(list(collection.find({}, {'price': 1, 'minimum nights': 1, 'number of reviews': 1, '_id': 0})))
    data['price'] = data['price'].apply(lambda x: float(x.replace('$', '').strip()))
    X = data[['minimum nights', 'number of reviews']]
    y = data['price']

    # Régression linéaire
    lr = LinearRegression()
    lr.fit(X, y)
    data['predicted_price'] = lr.predict(X)

    # Classification KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    data['price_category'] = pd.cut(data['price'], bins=3, labels=['Bas', 'Moyen', 'Élevé'])
    knn.fit(X, data['price_category'])
    data['predicted_category'] = knn.predict(X)

    return jsonify({'message': 'Machine learning appliqué'})

# Classification de document
@app.route('/classify/<id>', methods=['GET'])
def classify(id):
    document = collection.find_one({'_id': ObjectId(id)})
    if document:
        price = float(document.get('price', '0').replace('$', '').strip())
        if price > 100:
            classification = 'Cher'
        else:
            classification = 'Abordable'
        return jsonify({'id': str(document['_id']), 'NAME': document.get('NAME', ''), 'classification': classification})
    return jsonify({'error': 'Document non trouvé'}), 404

if __name__ == '__main__':
    app.run(debug=True)