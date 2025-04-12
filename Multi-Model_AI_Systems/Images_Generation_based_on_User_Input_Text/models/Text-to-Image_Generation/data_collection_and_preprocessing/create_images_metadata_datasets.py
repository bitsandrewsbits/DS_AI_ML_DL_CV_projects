# datasets creation for Stable Diffusion model(train, validation datasets)
import images_annotations_file_creation_for_img_dataset as ds_for_sd

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

def create_images_metadata_datasets(datasets_info: dict):
    for dataset_type in datasets_info:
        ds_creation = ds_for_sd.Images_Annotations_File_Creation_For_Images_Dataset(
            datasets_info[dataset_type][0],
            datasets_info[dataset_type][1],
            datasets_info[dataset_type][2],
        )
        ds_creation.main()

if __name__ == '__main__':
    create_images_metadata_datasets(datasets_info)
