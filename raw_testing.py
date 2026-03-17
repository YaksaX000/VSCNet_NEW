from build_dataset import FundusDataset 

dataset = FundusDataset(
    image_root="/media/storage/eye/TSGH/fundus_processed/images",
    json_path="/media/storage/eye/TSGH/Ophthal1117_processed_V3/matched_data/FundusImages_VF_match_60.json"
)

img, sem = dataset[0]

print(sem)
print(sem.shape)
print(sem.sum())