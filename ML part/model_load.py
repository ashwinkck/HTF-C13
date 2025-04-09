import pickle

# Load the model once when the server starts
with open(r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\Hackathons\HackToFuture sjec\ML part\smart_scheduler_model.pkl", "rb") as f:
    model = pickle.load(f)
