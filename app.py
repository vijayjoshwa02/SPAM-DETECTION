from flask import Flask, render_template, request, jsonify,url_for
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# File path to store user data
USER_DATA_FILE = "user_data.csv"

# Load existing user data if available
try:
    with open(USER_DATA_FILE, "r", newline='') as file:
        reader = csv.DictReader(file)
        users = {row['username']: {'fullname': row['fullname'], 'email': row['email'], 'password': row['password']} for row in reader}
except FileNotFoundError:
    # If the file doesn't exist, start with an empty dictionary
    users = {}

# Function to save user data to the CSV file
def save_user_data():
    with open(USER_DATA_FILE, "w", newline='') as file:
        fieldnames = ['username', 'fullname', 'email', 'password']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for username, user_data in users.items():
            writer.writerow({'username': username, **user_data})

# Load the dataset for spam detection
df = pd.read_table("Dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "sms_message"])

# Map labels to 0 and 1
df["label"] = df.label.map({"ham": 0, "spam": 1})

# Vectorize the text data
count_vector = CountVectorizer()
train = count_vector.fit_transform(df["sms_message"])
target = df["label"]

# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(train, target)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/detect')
def detect():
    return render_template('spamdetect.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Perform validation and authentication
    if username in users and users[username]['password'] == password:
        return jsonify({'success': True, 'redirect_url': url_for('detect')})
    else:
        return jsonify({'error': 'Invalid username or password.'}), 401

@app.route('/signup', methods=['POST'])
def signup():
    fullname = request.form['fullname']
    email = request.form['email']
    username = request.form['username']
    password = request.form['password']
    
    # Check if username or email is already registered
    if username in users or email in [user['email'] for user in users.values()]:
        return jsonify({'error': 'Username or email already exists.'}), 400
    # Save user data
    users[username] = {'fullname': fullname, 'email': email, 'password': password}
    save_user_data()  # Save user data to the file
    return jsonify({'message': 'Sign up successful!'})

@app.route('/detect_spam', methods=['POST'])
def detect_spam():
    text = request.form['text']
    predict_sms = naive_bayes.predict(count_vector.transform([text]))
    return render_template('spamdetect.html', prediction=predict_sms[0], text=text)

if __name__ == '__main__':
    app.run(debug=True)
