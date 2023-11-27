from flask import Flask, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask import request

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import load
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['SECRET_KEY'] = 'manchesterunited'
app.config['JWT_SECRET_KEY'] = 'manchesterunited'  
jwt = JWTManager(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../user_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    
def create_tables():
    with app.app_context():
        db.create_all()


@app.route("/", methods=["GET"])
def home():
    return "Welcome to our Solution!"
       
@app.route("/signup", methods=["POST"])
def signup():
    if request.method == 'POST':
        try:
            data = request.json
            print("Received data:", data)

            firstname = data['firstname']
            lastname = data['lastname']
            username = data['username']
            email = data['email']
            password = data['password']

            # Check if the email already exists
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already exists'}), 400

            new_user = User(
                firstname=firstname,
                lastname=lastname,
                username=username,
                email=email,
                password=password
            )

            db.session.add(new_user)
            db.session.commit()

            return "Registration Complete."
        except KeyError as ke:
            return jsonify({'error': f'Missing required field: {str(ke)}'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Unable to complete registration. Error: {e}'}), 500
        
# Sign-in endpoint
@app.route("/signin", methods=["POST"])
def signin():
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')

        user = User.query.filter_by(email=email, password=password).first()
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401

        access_token = create_access_token(identity=user.id)  # Create a JWT with user ID
        return jsonify({'access_token': access_token}), 200
        
 
# Sign-out endpoint
@app.route("/logout")
def logout():
    # Remove user info from the session or the token system
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

MODEL_FILE = "knn_model.pkl"
SCALER_FILE = "minmax_scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
COUNTRY_ENCODER_FILE = "country_encoder.pkl"
SEASON_ENCODER_FILE = "season_encoder.pkl"

COLUMNS_TO_SCALE = ["temperature", "humidity", "ph", "water availability"]
ENCODE_LABEL = "label"
ENCODE_COUNTRY = 'Country'

def preprocess_data(data, label_encoder, scaler, country_encoder):
    data[COLUMNS_TO_SCALE] = scaler.transform(data[COLUMNS_TO_SCALE])
    data[ENCODE_LABEL] = label_encoder.transform(data[ENCODE_LABEL])
    data[ENCODE_COUNTRY] = country_encoder.transform(data[ENCODE_COUNTRY])
    return data

def load_model():
    model = load(MODEL_FILE)
    scaler = load(SCALER_FILE)
    label_encoder = load(LABEL_ENCODER_FILE)
    country_encoder = load(COUNTRY_ENCODER_FILE)
    season_encoder = load(SEASON_ENCODER_FILE)
    
    
    return model, scaler, label_encoder, country_encoder, season_encoder

def get_predictions_info(predictions):
    return f" and harvest during {predictions[0]}."


model, scaler, label_encoder, country_encoder, season_encoder = load_model()

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    current_user_id = get_jwt_identity()
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Convert the data to a DataFrame with a single row
        data_df = pd.DataFrame(data, index=[0])
        processed_df = preprocess_data(data_df, label_encoder, scaler, country_encoder)
        # Make predictions using the loaded model
        predictions = model.predict(processed_df[['temperature', 'humidity', 'ph', 'water availability', 'label', 'Country']])
        
        pred = season_encoder.inverse_transform(pd.Series(predictions))[0]

        predictions_info = f"Expect to harvest during {pred} season."
        
        # Return the predictions as JSON
        response = jsonify({'predictions': predictions_info,
                            "current_user": current_user_id})
        return response, 200

    except Exception as e:
        # Return the error as JSON
        return jsonify({'error': str(e)}), 500



     
if __name__ == "__main__":
    create_tables()
    app.run(debug=True)