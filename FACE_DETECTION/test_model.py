from tensorflow.keras.models import load_model

# Load the saved model file
model = load_model('face_emotionModel.h5')

# Print a summary to confirm it’s loaded
model.summary()
