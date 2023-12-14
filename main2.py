import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load the data and perform preprocessing
db = pd.read_csv("dataset/Restaurants_Penang3.csv")
db['rating_ori'] = db['rating']
db['review_ori'] = db['review_count']
db = db.dropna()
cuisine_types = db['cuisine_type'].str.get_dummies(sep=',')
price_range = pd.get_dummies(db['price_level'], prefix='price')
db = pd.concat([db, cuisine_types, price_range], axis=1)
scaler = MinMaxScaler()
db[['rating', 'review_count']] = scaler.fit_transform(db[['rating', 'review_count']])

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/jp")
def Home_jp():
    return render_template("index_jp.html")

@app.route("/user-input")
def User_input():
    cuisineTypes = [
        'African', 'American', 'Arabic', 'Asian', 'Australian', 'Bar', 'Barbecue',
        'Belgian', 'Brazilian', 'British', 'Cafe', 'Campania', 'Cantonese',
        'Central Asian', 'Central European', 'Chinese', 'Contemporary', 'Deli',
        'Dessert', 'Diner', 'Dining bars', 'European', 'Fast food', 'French',
        'Fusion', 'Gastropub', 'German', 'Gluten Free Options', 'Greek', 'Grill',
        'Halal', 'Healthy', 'Indian', 'Indonesian', 'International', 'Irish',
        'Italian', 'Japanese', 'Japanese Fusion', 'Japanese sweets parlour',
        'Kaiseki', 'Korean', 'Latin', 'Lebanese', 'Malaysian', 'Mediterranean',
        'Mexican', 'Middle Eastern', 'Moroccan', 'Neapolitan', 'Nonya/Malaysian',
        'Pakistani', 'Philippine', 'Pizza', 'Portuguese', 'Pub', 'Quick Bites',
        'Seafood', 'Shanghai', 'Singaporean', 'Soups', 'Southwestern', 'Spanish',
        'Steakhouse', 'Street Food', 'Sushi', 'Swiss', 'Taiwanese', 'Thai',
        'Turkish', 'Tuscan', 'Vegan Options', 'Vegetarian Friendly', 'Vietnamese',
        'Wine Bar'
    ]

    priceLevels = ['1', '2', '3']

    return render_template("user_input.html", checkBox1=cuisineTypes, checkBox2=priceLevels)


# Define the prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():

    cuisine_type = request.form.getlist("Cuisine Type")
    price_level = request.form.getlist("Price Level")

    if not cuisine_type or not price_level:
        # Display alert if data is required
        return render_template("predict.html", alert_message1="Cuisine Type and Price Level are required.")

    user_fav_restaurant = pd.DataFrame({
        'price_level': [float(x) for x in price_level],
        #'price_level': [price_level],
        'cuisine_type': [cuisine_type],
    })
    
    user_cuisine_types =  user_fav_restaurant['cuisine_type'].apply(lambda x: ','.join(x)).str.replace(r'[\[\]]', '', regex=False).str.get_dummies(sep=',')
    user_price_range = pd.get_dummies(user_fav_restaurant['price_level'], prefix='price')

    # Filter restaurant_data based on cuisine_type and price_range
    filtered_restaurants = db[
        (db[user_cuisine_types.columns] == 1).any(axis=1) &
        (db[user_price_range.columns] == 1).any(axis=1)
    ]

    if filtered_restaurants.empty:
        # Display alert if no matching data found
        return render_template("predict.html", alert_message2="No matching data found.")
    
    # Make predictions on the filtered restaurants
    wide_inputs_filtered = np.concatenate((filtered_restaurants[cuisine_types.columns].values, filtered_restaurants[price_range.columns].values), axis=1)
    deep_inputs_filtered = np.array(filtered_restaurants[['rating', 'review_count']])

    predictions = model.predict([wide_inputs_filtered, deep_inputs_filtered])
    filtered_restaurants['prediction'] = predictions

    filtered_restaurants = filtered_restaurants.sort_values('prediction', ascending=False)
    sorted_restaurants = filtered_restaurants.head(8)
    sorted_restaurants['distance'] = None
    sorted_restaurants['duration'] = None
    
    # Prepare the response data
    response = {
        'top_recommendations': sorted_restaurants[['restaurant', 'cuisine_type', 'price_level', 'location', 'rating_ori', 'review_ori', 'Latitude', 'Longitude', 'url', 'image_url', 'distance', 'duration', 'prediction']].to_dict(orient='records')
    }

    return render_template("predict.html", predictions = response['top_recommendations'])

@app.route("/auto-generation")
def Auto_generation():

    return render_template("auto_gen.html")


@app.route("/predict-2", methods=['POST', 'GET'])
def Predict2():
    if request.method == 'POST':
        # liked_restaurants = []

        # cuisine_types = [cuisine.split(',') for cuisine in request.form.getlist('cuisineType')]
        # price_levels = [float(x) for x in request.form.getlist('priceLevel')]

        # liked_restaurant = {
        #     'cuisineType': cuisine_types,
        #     'priceLevel': price_levels
        # }
        # liked_restaurants.append(liked_restaurant)

        # return render_template('predict2.html', liked_restaurants=liked_restaurants)


        cuisine_type = [cuisine.split(',') for cuisine in request.form.getlist('cuisineType')]
        price_level = [float(x) for x in request.form.getlist('priceLevel')]

        if not cuisine_type or not price_level:
        # Display alert if data is required
            return render_template("predict2.html", alert_message1="No liked record found.")

        user_fav_restaurant = pd.DataFrame({
            'price_level': price_level,
            'cuisine_type': cuisine_type,
        })

        # user_cuisine_types = pd.get_dummies(user_fav_restaurant['cuisine_type'])
        user_cuisine_types =  user_fav_restaurant['cuisine_type'].apply(lambda x: ','.join(x)).str.replace(r'[\[\]]', '', regex=False).str.get_dummies(sep=',')
        user_price_range = pd.get_dummies(user_fav_restaurant['price_level'], prefix='price')

        # Filter restaurant_data based on cuisine_type and price_range
        filtered_restaurants = db[
            (db[user_cuisine_types.columns] == 1).any(axis=1) &
            (db[user_price_range.columns] == 1).any(axis=1)
        ]

        # Make predictions on the filtered restaurants
        wide_inputs_filtered = np.concatenate((filtered_restaurants[cuisine_types.columns].values, filtered_restaurants[price_range.columns].values), axis=1)
        deep_inputs_filtered = np.array(filtered_restaurants[['rating', 'review_count']])

        predictions = model.predict([wide_inputs_filtered, deep_inputs_filtered])
        filtered_restaurants['prediction'] = predictions

        filtered_restaurants = filtered_restaurants.sort_values('prediction', ascending=False)
        sorted_restaurants = filtered_restaurants.head(8)
        sorted_restaurants['distance'] = None
        sorted_restaurants['duration'] = None
        
        # Prepare the response data
        response = {
            'top_recommendations': sorted_restaurants[['restaurant', 'cuisine_type', 'price_level', 'location', 'rating_ori', 'review_ori', 'Latitude', 'Longitude', 'url', 'image_url', 'distance', 'duration', 'prediction']].to_dict(orient='records')
        }

        return render_template("predict2.html", predictions2 = response['top_recommendations'])


    # Handle the case when form data is not available
    return render_template('predict2.html', error_message='Form data not available')



########################################################################
# For food recommendation
df = pd.read_csv('dataset/food.csv')  
def text_cleaning(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text
df['Describe'] = df['Describe'].apply(text_cleaning)
def create_mix(x):
    return x['Name'] + " " + x['C_Type'] + " " + x['Veg_Non'] + " " + x['Describe']
df['mix'] = df.apply(create_mix, axis=1).str.lower()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['mix'])


@app.route("/food-input")
def Food_input():
    print('hi')
    return render_template("food_input.html")

@app.route("/predict-3", methods=['POST', 'GET'])
def Predict3():
    if request.method == 'POST':
        
        user_keywords = request.form.get('keywords', '')

        user_keywords_vector = tfidf.transform([user_keywords])

        # Calculate similarity scores
        cosine_sim = cosine_similarity(user_keywords_vector, tfidf_matrix)

        # List top 5 recommended foods
        top_indices = cosine_sim.argsort()[0][::-1][:6]
        recommended_food = df.iloc[top_indices][['Name']].to_dict(orient='list')
        recommended_scores = cosine_sim[0][top_indices].tolist()
        
        # Combine food names with scores
        recommendations = [{'Name': food, 'Score': score} for food, score in zip(recommended_food['Name'], recommended_scores)]
        # print(recommendations)
        return render_template("predict3.html", user_keywords=user_keywords, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
