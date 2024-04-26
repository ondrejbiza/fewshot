from src import utils
from src.utils import CanonPart, CanonPartMetadata
import pickle

model = pickle.load(open("./part_based_warp_models/branch_20240412-042732", "rb"))
model_dict = CanonPart.to_dict(model)
pickle.dump(
    model_dict, open("./part_based_warp_models/branch_dict_20240412-042732", "wb")
)
new_model = CanonPart.from_pickle(
    "./part_based_warp_models/branch_dict_20240412-042732"
)
