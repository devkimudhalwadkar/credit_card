import streamlit as st

# Title of the Streamlit app
st.title("Credit Card Fraud Detection App")

# Section to upload the trained model (.h5 file)
st.header("Upload Your Trained Model (.h5)")
model_file = st.file_uploader("Choose a Keras model file", type=["h5"])

if model_file is not None:
    # Save the uploaded model file temporarily
    with open("credit_card_fraud_detect_NN_modelemp_model.h5", "wb") as f:
        f.write(model_file.getbuffer())
    
    try:
        model = load_model("temp_model.h5")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # Section to upload CSV data for prediction
    st.header("Upload CSV Data for Prediction")
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if csv_file is not None:
        try:
            data = pd.read_csv(csv_file)
            st.write("### Data Preview", data.head())
            
            # OPTIONAL: Preprocess data as required by your model
            # For example, if your model expects scaled features, perform scaling here.
            # If a column like 'Class' exists, you may want to drop it:
            if "Class" in data.columns:
                data_features = data.drop(columns=["Class"])
            else:
                data_features = data

            # Generate predictions using the model
            predictions = model.predict(data_features)
            
            # If your model is for binary classification, you might want to convert
            # probabilities into binary class labels (using 0.5 as threshold)
            if predictions.shape[1] == 1:
                predictions = (predictions > 0.5).astype("int32")
            else:
                predictions = np.argmax(predictions, axis=1)
            
            # Append predictions to the original dataframe
            data["Prediction"] = predictions
            st.write("### Predictions", data)
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
else:
    st.info("Please upload your model file to start.")
