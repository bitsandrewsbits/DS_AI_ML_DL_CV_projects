# datasets creation for Stable Diffusion model(train, validation datasets)
import dataset_creation_for_Stable_Diffusion_model as ds_for_sd

datasets_info = {
    "TRAIN": [
        'train2017',
        'instances_train2017.json',
        'captions_train2017.json'],
    "VALIDATION": [
        'validation2017_for_fine-tune',
        'instances_val2017.json',
        'captions_val2017.json']
}

def create_datasets(datasets_info: dict):
    for dataset_type in datasets_info:
        ds_creation = ds_for_sd.Dataset_Creating_for_Stable_Diffusion(
            datasets_info[dataset_type][0],
            datasets_info[dataset_type][1],
            datasets_info[dataset_type][2],
        )
        ds_creation.main(dataset_type)

if __name__ == '__main__':
    create_datasets(datasets_info)
