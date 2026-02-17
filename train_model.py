import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# 1. Load the data
print("Loading data...")
df = pd.read_csv('admission_data.csv')

# 2. Clean the data (Remove 'Serial No.' as it's not useful)
# Note: Check your CSV column names. Sometimes there is a space at the end like 'Chance of Admit '
df.columns = df.columns.str.strip() # Removes accidental spaces from column names
df = df.drop(columns=['Serial No.'])

# 3. Define X (Inputs) and y (Output)
X = df.drop(columns=['Chance of Admit'])
y = df['Chance of Admit']

# 4. Split data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression Model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Check Accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# 7. Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Success! Model saved as 'model.pkl'")