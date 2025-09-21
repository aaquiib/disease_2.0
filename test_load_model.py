import tensorflow as tf

def test_load_model():
    model_path = "model/7"  # Update if needed
    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    test_load_model()
