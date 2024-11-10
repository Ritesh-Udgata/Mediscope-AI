import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import ast
import itertools
import os
import logging
from datetime import datetime
import google.generativeai as genai


import json
from flask import Flask, request, redirect, url_for, render_template, flash, session,jsonify
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
# Initialize Google API key
GOOGLE_API_KEY = 'your_api_key'
genai.configure(api_key=GOOGLE_API_KEY)
CONVERSATION_FILE = 'conversation.json'

def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, 'r') as file:
            return json.load(file)
    return []
def save_conversation(conversation):
    with open(CONVERSATION_FILE, 'w') as file:
        json.dump(conversation, file)

# Initialize model
model = genai.GenerativeModel('gemini-pro')

# JSON file path for storing user data
USER_DATA_FILE = 'users.json'

# Helper function to load user data from JSON file
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, 'r') as file:
        return json.load(file)

# Helper function to save user data to JSON file
def save_users(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

# Sign Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Load existing users
        users = load_users()

        # Check if username already exists
        if username in users:
            flash("Username already exists. Please choose a different one.")
            return redirect(url_for('signup'))

        # Hash the password and save the new user
        users[username] = generate_password_hash(password)
        save_users(users)

        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Load users
        users = load_users()

        # Authenticate user
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            flash("Login successful!")
            return redirect(url_for('main_page'))
        
        flash("Invalid username or password.")
        return redirect(url_for('login'))
    
    return render_template('login.html')

# Main Page Route (Protected)
@app.route('/main')
def main_page():
    if 'user' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))
    return redirect(url_for('index'))

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))


# Add this route in your Flask app
@app.route('/chat')
def chat():
    conversation = load_conversation()  # Load conversation history if you want to display it
    return render_template('chat.html', conversation=conversation)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Load conversation history
    conversation = load_conversation()

    # Add user message to conversation history
    conversation.append({'role': 'user', 'message': user_message})

    # Generate chatbot response
    chat = model.start_chat(history=conversation)
    response = chat.send_message(user_message, stream=False)

    # Add bot response to conversation history
    bot_message = response.text
    conversation.append({'role': 'bot', 'message': bot_message})

    # Save updated conversation to JSON file
    save_conversation(conversation)

    # Return bot's response
    return jsonify({'response': bot_message})

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# List of symptoms (377 symptoms)
symptoms = ['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness']  # ... rest of symptoms list

def validate_input_symptoms(symptoms_list):
    """Validate that input symptoms are in the allowed list"""
    try:
        if not isinstance(symptoms_list, list):
            raise ValueError("Input symptoms must be a list")
        if not all(isinstance(s, (int, float)) for s in symptoms_list):
            raise ValueError("All symptoms must be numeric (0 or 1)")
        if len(symptoms_list) != 377:
            raise ValueError(f"Expected 377 symptoms, got {len(symptoms_list)}")
        return True
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        raise

def load_model_and_encoder():
    """Load the model and label encoder with enhanced error handling"""
    try:
        model_path = os.path.join('model', 'Model_4_better.h5')
        encoder_path = os.path.join('model', 'label_encoder.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found at: {encoder_path}")
            
        # Load model and encoder
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info("Successfully loaded model and encoder")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading model or encoder: {str(e)}")
        raise

def load_drug_data():
    """Load drug-related CSV files with error handling"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, 'data')
        
        # Load CSV files
        finaldiseases = pd.read_csv(os.path.join(data_dir, 'finaldiseases.csv'))
        druginteraction = pd.read_csv(os.path.join(data_dir, 'druginteractionsfinal.csv'))
        singledrugeffect = pd.read_csv(os.path.join(data_dir, 'singledrugsideeffect.csv'))
        finaldiseases = finaldiseases.fillna('')  # Or dropna() or use mean/median
        druginteraction = druginteraction.fillna('')
        singledrugeffect = singledrugeffect.fillna('')
        
        # Validate data
        if finaldiseases.empty or druginteraction.empty or singledrugeffect.empty:
            raise ValueError("One or more CSV files are empty")
            
        logger.info("Successfully loaded drug data")
        return finaldiseases, singledrugeffect, druginteraction
    except Exception as e:
        logger.error(f"Error loading drug data: {str(e)}")
        raise

def get_unique_medicines(prediction_list, finaldiseases):
    """Get unique medicines for predicted diseases"""
    try:
        results = []
        for disease in prediction_list:
            drug_info = finaldiseases[finaldiseases['disease'] == disease]['drug'].values
            if len(drug_info) > 0:
                actual_drug_list = ast.literal_eval(drug_info[0])
                if isinstance(actual_drug_list, list):
                    unique_drugs = list(set(actual_drug_list))
                    results.append((disease, unique_drugs))
        return results
    except Exception as e:
        logger.error(f"Error getting unique medicines: {str(e)}")
        raise

def get_first_5_medicines(result_list):
    """Get first 5 medicines for each disease"""
    try:
        return [(disease, medicines[:5]) for disease, medicines in result_list]
    except Exception as e:
        logger.error(f"Error getting first 5 medicines: {str(e)}")
        raise

def get_side_effects_for_medicines(all_medicines, drug_df):
    """Get side effects for medicines"""
    try:
        medicine_side_effects = {}
        for medicine in all_medicines:
            row = drug_df[drug_df['drug_name'].str.lower() == medicine.lower()]
            if not row.empty:
                medicine_side_effects[medicine] = row.iloc[0]['side_effects']
            else:
                row = drug_df[drug_df['generic_name'].str.lower() == medicine.lower()]
                if not row.empty:
                    medicine_side_effects[medicine] = row.iloc[0]['side_effects']
        return medicine_side_effects
    except Exception as e:
        logger.error(f"Error getting side effects: {str(e)}")
        raise

def get_interactions_for_pairs(all_medicines, interaction_df):
    """Get drug interactions for medicine pairs"""
    try:
        interaction_dict = {}
        for drug1, drug2 in itertools.combinations(all_medicines, 2):
            pair_key = f"{drug1}|||{drug2}"
            
            pair1 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug1.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug2.lower())
            ]
            pair2 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug2.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug1.lower())
            ]
            
            if not pair1.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug1,
                    'drug_b': drug2,
                    'interaction': pair1.iloc[0]['Interaction'],
                    'risk_level': pair1.iloc[0]['Risk_Level']
                }
            elif not pair2.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug2,
                    'drug_b': drug1,
                    'interaction': pair2.iloc[0]['Interaction'],
                    'risk_level': pair2.iloc[0]['Risk_Level']
                }
        return interaction_dict
    except Exception as e:
        logger.error(f"Error getting drug interactions: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page"""
    try:
        return render_template('index.html', symptoms_list=symptoms)
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return jsonify({"error": "Error loading page"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Log request
        logger.info("Received prediction request")
        
        # Get and validate input data
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({"error": "Missing symptoms data"}), 400
            
        # Validate input format
        input_symptoms = np.array([data['symptoms']])
        validate_input_symptoms(data['symptoms'])

        # Load required components
        model, label_encoder = load_model_and_encoder()
        finaldiseases, singledrugeffect, druginteraction = load_drug_data()

        # Make prediction
        predicted_probabilities = model.predict(input_symptoms)
        sorted_indices = np.argsort(predicted_probabilities[0])[::-1]
        max_prob = predicted_probabilities[0][sorted_indices[0]]
        
        # Process predictions
        prediction_list = []
        prediction_list_prob = []
        extra_prediction = None
        extra_prediction_prob = None
        
        predicted_diseases = label_encoder.inverse_transform(sorted_indices)
        
        # Handle predictions based on probability threshold
        if max_prob < 0.5:
            first_prob = predicted_probabilities[0][sorted_indices[0]]
            prediction_list.append(predicted_diseases[0])
            prediction_list_prob.append(float(first_prob))
            
            for i in range(1, min(3, len(sorted_indices))):
                next_prob = predicted_probabilities[0][sorted_indices[i]]
                if first_prob - next_prob <= 0.2:
                    prediction_list.append(predicted_diseases[i])
                    prediction_list_prob.append(float(next_prob))
        else:
            first_prob = predicted_probabilities[0][sorted_indices[0]]
            prediction_list.append(predicted_diseases[0])
            prediction_list_prob.append(float(first_prob))
            
            for i in range(1, len(sorted_indices)):
                next_prob = predicted_probabilities[0][sorted_indices[i]]
                if first_prob - next_prob <= 0.2:
                    prediction_list.append(predicted_diseases[i])
                    prediction_list_prob.append(float(next_prob))
                else:
                    break
        
        # Get extra prediction
        extra_idx = len(prediction_list)
        if extra_idx < len(predicted_diseases):
            extra_prediction = predicted_diseases[extra_idx]
            extra_prediction_prob = float(predicted_probabilities[0][sorted_indices[extra_idx]])

        # Get medicine recommendations
        result = get_unique_medicines(prediction_list, finaldiseases)
        result_extra = get_unique_medicines([extra_prediction] if extra_prediction else [], finaldiseases)
        
        result_first5 = get_first_5_medicines(result)
        result_extra_first5 = get_first_5_medicines(result_extra)

        # Get all medicines
        all_medicines = []
        for disease, medicines in result_first5 + result_extra_first5:
            all_medicines.extend(medicines)
        all_medicines = list(set(all_medicines))

        # Get side effects and interactions
        side_effects_dict = get_side_effects_for_medicines(all_medicines, singledrugeffect)
        interaction_results = get_interactions_for_pairs(all_medicines, druginteraction)

        # Format predictions
        formatted_predictions = [
            {
                "disease": disease,
                "probability": round(float(prob), 4)
            } 
            for disease, prob in zip(prediction_list, prediction_list_prob)
        ]

        # Format extra prediction
        formatted_extra = None
        if extra_prediction:
            formatted_extra = {
                "disease": extra_prediction,
                "probability": round(float(extra_prediction_prob), 4)
            }

        # Prepare response
        response = {
            "predictions": formatted_predictions,
            "extra_prediction": formatted_extra,
            "medicines": {
                "main": [
                    {
                        "disease": disease,
                        "medications": medicines
                    }
                    for disease, medicines in result_first5
                ],
                "extra": [
                    {
                        "disease": disease,
                        "medications": medicines
                    }
                    for disease, medicines in result_extra_first5
                ]
            },
            "side_effects": side_effects_dict,
            "interactions": interaction_results
        }
        
        logger.info("Successfully generated prediction response")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    try:
        # Test model loading
        model, encoder = load_model_and_encoder()
        logger.info("Model loaded successfully")
        
        # Test data loading
        finaldiseases, singledrugeffect, druginteraction = load_drug_data()
        logger.info("Data files loaded successfully")
        
        # Print some basic stats
        logger.info(f"Number of diseases in data: {len(finaldiseases)}")
        logger.info(f"Number of drug interactions: {len(druginteraction)}")
        logger.info(f"Number of side effects: {len(singledrugeffect)}")
        
        app.run(debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)

if __name__ == '__main__':
    app.run(debug=True)
