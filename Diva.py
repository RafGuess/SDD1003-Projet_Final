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
    room_type = request.args.get('room_type')
    min_nights = request.args.get('min_nights')
    max_price = request.args.get('max_price')

    query = {}
    if room_type:
        query['room type'] = room_type
    if min_nights:
        query['minimum nights'] = {'$lte': int(min_nights)}
    if max_price:
        query['price'] = {'$lte': float(max_price)}

    results = list(collection.find(query, {'_id': 0, 'NAME': 1, 'price': 1, 'room type': 1, 'minimum nights': 1}))
    return jsonify(results)


# Fonctionnalité de mise à jour
@app.route('/update/<id>', methods=['PUT'])
def update(id):
    new_data = request.json
    result = collection.update_one({'_id': ObjectId(id)}, {'$set': new_data})
    if result.modified_count:
        return jsonify({'message': 'Mise à jour réussie'})
    return jsonify({'error': 'Échec de la mise à jour'}), 400


# Fonctionnalité de suppression
@app.route('/delete/<id>', methods=['DELETE'])
def delete(id):
    result = collection.delete_one({'_id': ObjectId(id)})
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
        price = document.get('price', 0)
        if price > 100:
            classification = 'Cher'
        else:
            classification = 'Abordable'
        return jsonify({'id': str(document['_id']), 'NAME': document.get('NAME', ''), 'classification': classification})
    return jsonify({'error': 'Document non trouvé'}), 404


if __name__ == '__main__':
    app.run(debug=True)