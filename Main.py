from flask import Flask, request, jsonify, render_template
import base64
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import io
import numpy as np
from sklearn.tree import DecisionTreeRegressor


# Initialisation de l'application Flask
app = Flask(__name__)

# Connexion à MongoDB
client = MongoClient('mongodb+srv://aguess1874:Alex.1874@cluster0.mr9n5.mongodb.net/')
db = client['AirBnB']
collection = db['AirBnB']

'''
Liste des champs:
'_id', 'id', 'NAME', 'host id', 'host_identity_verified', 'host name', 'neighbourhood group', 'neighbourhood',
'lat', 'long', 'country', 'country code', 'instant_bookable', 'cancellation_policy', 'room type', 'Construction year',
'price', 'service fee', 'minimum nights', 'number of reviews', 'last review', 'reviews per month', 'review rate number',
'calculated host listings count', 'availability 365', 'house_rules'
'''

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Fonctionnalité de recherche
@app.route('/search', methods=['GET'])
def search():
    # Récupération des paramètres de la requête
    min_nights = request.args.get('min_nights')
    neighbourhood_group = request.args.get('neighbourhood_group')
    instant_bookable = request.args.get('instant_bookable')

    # Construction de la requête MongoDB
    query = {'price': {'$exists': True, '$ne': ''}}
    if min_nights:
        query['minimum nights'] = {'$gte': int(min_nights)}
    if neighbourhood_group:
        query['neighbourhood group'] = neighbourhood_group
    if instant_bookable:
        query['instant_bookable'] = True

    # Exécution de la requête et récupération des résultats
    results = list(collection.find(query, {'_id': 0, 'NAME': 1, 'price': 1, 'minimum nights': 1, 'instant_bookable': 1, 'neighbourhood group': 1}).sort('price', 1))
    return render_template('search_results.html', results=results)

# Fonctionnalité de mise à jour
@app.route('/search_update', methods=['GET'])
def search_update():
    # Récupération du nom pour la mise à jour
    name = request.args.get('name')
    query = {'price': {'$exists': True, '$ne': ''}}
    if name:
        query['NAME'] = name

    # Recherche du document correspondant
    result = collection.find_one(query, {'_id': 1, 'NAME': 1, 'price': 1, 'minimum nights': 1, 'instant_bookable': 1, 'neighbourhood group': 1})
    if result:
        return render_template('update.html', result=result)
    return jsonify({'error': 'Document non trouvé'}), 404

# Route pour effectuer la mise à jour
@app.route('/update', methods=['POST'])
def update():
    # Récupération des données du formulaire
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

    # Mise à jour du document dans MongoDB
    result = collection.update_one({'_id': ObjectId(document_id)}, {'$set': new_data})
    if result.modified_count:
        return jsonify({'message': 'Mise à jour réussie'})
    return jsonify({'error': 'Échec de la mise à jour'}), 400

# Route pour la suppression d'un document
@app.route('/delete', methods=['POST'])
def delete():
    # Récupération du nom pour la suppression
    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Le paramètre name est manquant'}), 400

    # Suppression du document dans MongoDB
    result = collection.delete_one({'NAME': name})
    if result.deleted_count:
        return jsonify({'message': 'Suppression réussie'})
    return jsonify({'error': 'Échec de la suppression, document non trouvé'}), 404

# Fonctionnalité de génération de graphiques
@app.route('/graphs')
def graphs():
    # Chargement des données depuis MongoDB
    data = pd.DataFrame(list(collection.find({}, {'room type': 1, 'minimum nights': 1, 'number of reviews': 1, 'date': 1, 'instant_bookable': 1, 'neighbourhood group': 1, '_id': 0})))

    # Distribution des types de chambres
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    data['room type'].value_counts().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution des types de chambres', fontsize=16)
    ax1.set_xlabel('Type de chambre', fontsize=14)
    ax1.set_ylabel('Nombre', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png')
    img1.seek(0)
    graph1_url = base64.b64encode(img1.getvalue()).decode()

    # Distribution des nuits minimums
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    data['minimum nights'].plot(kind='hist', bins=50, ax=ax2, color='lightgreen', edgecolor='black')
    ax2.set_title('Distribution des nuits minimums', fontsize=16)
    ax2.set_xlabel('Nuits minimums', fontsize=14)
    ax2.set_ylabel('Fréquence', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(-10, 500)
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    graph2_url = base64.b64encode(img2.getvalue()).decode()

    # Croisement instant_bookable et neighbourhood_group
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    cross_tab = pd.crosstab(data['neighbourhood group'], data['instant_bookable'])
    cross_tab.plot(kind='bar', ax=ax3, color=['lightcoral', 'lightblue'], edgecolor='black')
    ax3.set_title('Instant Bookable vs Neighbourhood Group', fontsize=16)
    ax3.set_xlabel('Neighbourhood Group', fontsize=14)
    ax3.set_ylabel('Count', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0, ha='right')
    img3 = io.BytesIO()
    fig3.savefig(img3, format='png')
    img3.seek(0)
    graph3_url = base64.b64encode(img3.getvalue()).decode()

    return render_template('graphs.html', graph1_url=graph1_url, graph2_url=graph2_url, graph3_url=graph3_url)

# Algorithmes de machine learning
@app.route('/ml')
def ml():
    # Chargement des données depuis MongoDB
    data = pd.DataFrame(list(collection.find({}, {'minimum nights': 1, 'number of reviews': 1, '_id': 0})))

    # Prétraitement des données
    data = data.dropna()
    X = data[['minimum nights', 'number of reviews']]
    y = np.random.rand(len(data))  # Variable cible factice pour la démonstration

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)

    # Évaluation du modèle
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Visualisation des résultats
    plt.scatter(y_test, predictions)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Valeurs réelles vs Valeurs prédites')
    plt.show()

    return jsonify({
        'predictions': predictions.tolist(),
        'mse': mse,
        'r2': r2
    })




# Algorithmes de machine learning - KNN
@app.route('/ml_knn')
def ml_knn():
    # Chargement des données depuis MongoDB
    data = pd.DataFrame(list(collection.find({}, {'minimum nights': 1, 'number of reviews': 1, '_id': 0})))

    # Prétraitement des données
    data = data.dropna()
    X = data[['minimum nights', 'number of reviews']]
    y = np.random.rand(len(data))  # Variable cible factice pour la démonstration

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle KNN
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Prédictions
    knn_predictions = knn_model.predict(X_test)

    # Évaluation du modèle
    knn_mse = mean_squared_error(y_test, knn_predictions)
    knn_r2 = r2_score(y_test, knn_predictions)

    # Visualisation des résultats
    plt.scatter(y_test, knn_predictions)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Valeurs réelles vs Valeurs prédites (KNN)')
    plt.show()

    return jsonify({
        'predictions': knn_predictions.tolist(),
        'mse': knn_mse,
        'r2': knn_r2
    })


# Algorithmes de machine learning - Decision Tree Regression
@app.route('/ml_decision_tree')
def ml_decision_tree():
    # Chargement des données depuis MongoDB
    data = pd.DataFrame(list(collection.find({}, {'minimum nights': 1, 'number of reviews': 1, '_id': 0})))

    # Prétraitement des données
    data = data.dropna()
    X = data[['minimum nights', 'number of reviews']]
    y = np.random.rand(len(data))  # Variable cible factice pour la démonstration

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle Decision Tree Regression
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)

    # Prédictions
    dt_predictions = dt_model.predict(X_test)

    # Évaluation du modèle
    dt_mse = mean_squared_error(y_test, dt_predictions)
    dt_r2 = r2_score(y_test, dt_predictions)

    # Visualisation des résultats
    plt.scatter(y_test, dt_predictions)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Valeurs réelles vs Valeurs prédites (Decision Tree)')
    plt.show()

    return jsonify({
        'predictions': dt_predictions.tolist(),
        'mse': dt_mse,
        'r2': dt_r2
    })

# Classification de document
@app.route('/classify', methods=['GET'])
def classify():
    # Récupération du nom du document à classifier
    name = request.args.get('name')
    if not name:
        return jsonify({'error': 'Le paramètre name est manquant'}), 400

    # Recherche du document par nom
    document = collection.find_one({'NAME': name})
    if document:
        # Classification basée sur le nombre de reviews
        number_of_reviews = document.get('number of reviews', 0)
        if number_of_reviews > 50:
            classification = 'Populaire'
        else:
            classification = 'Moins populaire'
        return jsonify({'name': document.get('NAME', ''), 'classification': classification})
    return jsonify({'error': 'Document non trouvé'}), 404

# Démarrage de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)