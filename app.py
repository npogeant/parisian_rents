
from flask import Flask, render_template, request, redirect, jsonify, make_response, url_for
import json
from sqlalchemy import create_engine, text, Table, Column, MetaData, Integer, Computed
import pandas as pd
from cachetools import cached, TTLCache
from config import Config

import pickle
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

# Initiate Flask Application
app = Flask(__name__, static_folder='templates')

cache = TTLCache(maxsize=100, ttl=60)

# Routing to home url
@app.route('/')
def index():

    return render_template('index.html')

#  Routing that get the prediction from the form responses
@app.route('/get_prediction')
def calculate_result():

    # Preprocessing function
    def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
        ''' Create feature value vectors and one hot encode categorical
            columns.

            Args :
            - df : A Dataframe
            - dv : A DictVectorizer object (from Scikit Learn)
            - fit_dv : True or False, if True, it fits the dv with 
                        the specified Dataframe.
        '''
        categorical = ['Epoque de construction', 'Type de location']
        numerical = ['Secteurs géographiques', 'Numéro du quartier',
                     'Nombre de pièces principales']
        dicts = df[categorical + numerical].to_dict(orient='records')
        if fit_dv:
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
        return X, dv

    # Import the model and the dictvectorizer
    final_xgbregressor_model_path = 'final_xgbregressor_model.model'
    preprocessor_path = 'preprocessor.b'
    dv = pickle.load(open(f"utils/model/{preprocessor_path}", "rb"))
    xbg_regressor = xgb.XGBRegressor()
    xbg_regressor.load_model(f"utils/model/{final_xgbregressor_model_path}")

    # Populate the features
    neighborhood = request.args.get('neighborhood')
    period = request.args.get('period')
    main_rooms = int(request.args.get('main_rooms'))
    type = request.args.get('type')
    area = int(request.args.get('area'))

    # As the model learned with french data, we have to 
    # change the feature variable with the french one
    type_dict = {'After 1990': 'Apres 1990',
                 'Before 1946': 'Avant 1946',
                 '1971-1990': '1971-1990',
                 '1946-1970': '1946-1970',
                 'Furnished': 'meublé',
                 'Unfurnished': 'non meublé',
                 'Amérique': '13',
                 'Archives': '4',
                 'Arsenal': '2',
                 'Arts-et-Metiers': '4',
                 'Auteuil': '7',
                 'Batignolles': '10',
                 'Bel-Air': '14',
                 'Belleville': '11',
                 'Bercy': '14',
                 'Bonne-Nouvelle': '4',
                 'Chaillot': '3',
                 'Champs-Elysées': '2',
                 'Charonne': '13',
                 "Chaussée-d'Antin": '2',
                 'Clignancourt': '9',
                 'Combat': '14',
                 'Croulebarbe': '5',
                 'Ecole-Militaire': '1',
                 'Enfants-Rouges': '4',
                 'Epinettes': '11',
                 'Europe': '3',
                 'Faubourg-Montmartre': '5',
                 'Faubourg-du-Roule': '2',
                 'Folie-Méricourt': '11',
                 'Gaillon': '2',
                 'Gare': '13',
                 "Goutte-d'Or": '11',
                 'Grandes-Carrières': '9',
                 'Grenelle': '7',
                 'Gros-Caillou': '1',
                 'Halles': '5',
                 'Hôpital-Saint-Louis': '11',
                 'Invalides': '1',
                 'Jardin-des-Plantes': '10',
                 'Javel 15Art': '7',
                 'La Chapelle': '13',
                 'Madeleine': '2',
                 'Mail': '4',
                 'Maison-Blanche': '12',
                 'Monnaie': '2',
                 'Montparnasse': '5',
                 'Muette': '3',
                 'Necker': '6',
                 'Notre-Dame': '2',
                 'Notre-Dame-des-Champs': '1',
                 'Odeon': '2',
                 'Palais-Royal': '2',
                 'Parc-de-Montsouris': '11',
                 'Petit-Montrouge': '10',
                 'Picpus': '9',
                 'Place-Vendôme': '2',
                 'Plaine de Monceaux': '6',
                 'Plaisance': '12',
                 'Pont-de-Flandre': '13',
                 'Porte-Dauphine': '3',
                 'Porte-Saint-Denis': '5',
                 'Porte-Saint-Martin': '11',
                 'Père-Lachaise': '14',
                 'Quinze-Vingts': '11',
                 'Rochechouart': '5',
                 'Roquette': '11',
                 'Saint-Ambroise': '10',
                 'Saint-Fargeau': '13',
                 'Saint-Georges': '5',
                 'Saint-Germain-des-Prés': '2',
                 'Saint-Gervais': '4',
                 'Saint-Lambert': '8',
                 'Saint-Merri': '2',
                 "Saint-Thomas-d'Aquin": '1',
                 'Saint-Victor': '4',
                 'Saint-Vincent-de-Paul': '5',
                 'Sainte-Avoie': '4',
                 'Sainte-Marguerite': '10',
                 'Salpêtrière': '10',
                 'Sorbonne': '4',
                 "St-Germain-l'Auxerrois": '2',
                 'Ternes': '6',
                 'Val-de-Grace': '4',
                 'Villette': '13',
                 'Vivienne': '4'
                 }

    # Features name
    columns = ['Epoque de construction',
               'Type de location',
               'Secteurs géographiques',
               'Numéro du quartier',
               'Nombre de pièces principales']
    # Values
    values = [[type_dict.get(period),
              type_dict.get(type),
              neighborhood,
              type_dict.get(neighborhood),
              main_rooms]]
    
    # Create a Dataframe with the response
    df = pd.DataFrame(data=values,
                      columns=columns)

    # Initialize the inputs and the dv
    X, dv = preprocess(df=df, dv=dv)
    
    # Get the prediction
    pred = xbg_regressor.predict(X)
    
    # Multiply the prediction with the area value from the form
    rent = round(pred[0]*area)

    # The message that will be displayed in the html
    message = f"The predicted rent for an apartment located at <mark>{neighborhood}</mark>,\
                in a period <mark>{period}</mark>, with <mark>{main_rooms}</mark> main rooms\
                and <mark>{type}</mark> is : <mark2>{rent} €</mark2> 🥳 "
                
    return jsonify({"result": message})


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True, port=5000)
