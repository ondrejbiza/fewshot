from src import utils
from src.utils import CanonPart, CanonPartMetadata
import pickle

model = pickle.load(open("./part_based_warp_models/whole_bowl_20240425-235458_10", "rb"))
model_dict = CanonPart.to_dict(model)
pickle.dump(
    model_dict, open("./part_based_warp_models/whole_bowl_dict_20240425-235458_10", "wb")
)

# model = pickle.load(open("./part_based_warp_models/body_20241031-012841_5", "rb"))
# model_dict = CanonPart.to_dict(model)
# pickle.dump(
#     model_dict, open("./part_based_warp_models/body_dict_20241031-012841_5", "wb")
# )

# model = pickle.load(open("./part_based_warp_models/lid_20241031-012841_5", "rb"))
# model_dict = CanonPart.to_dict(model)
# pickle.dump(
#     model_dict, open("./part_based_warp_models/lid_dict_20241031-012841_5", "wb")
# )

# model = pickle.load(open("./part_based_warp_models/spout_20241031-012841_5", "rb"))
# model_dict = CanonPart.to_dict(model)
# pickle.dump(
#     model_dict, open("./part_based_warp_models/spout_dict_20241031-012841_5", "wb")
#)
# new_model = CanonPart.from_pickle(
#     "./part_based_warp_models/handle_dict_20240430-034748_5"
# )
