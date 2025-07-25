# Steps for data downloading, processing COCO dataset, SD model fine-tuning.

# current dir should be - data_collection_and_preprocessing

#1) Create 4 dirs tree - data/train2017/annotations, data/datasets_for_fine-tune:
mkdir -p data/train2017/annotations
mkdir -p data/validation2017_for_fine-tune

#2) install axel util for downloading files.
sudo apt install axel

#3) download files via axel(3 files - train2017.zip, val2017.zip, annotations_trainval2017.zip):
axel -o data/train2017/train2017.zip http://images.cocodataset.org/zips/train2017.zip
axel -o data/validation2017_for_fine-tune/val2017.zip http://images.cocodataset.org/zips/val2017.zip
axel -o data/validation2017_for_fine-tune/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip

#4) unzip files with respect to their dirs:
unzip -q data/train2017/train2017.zip -d data/train2017
unzip -q data/validation2017_for_fine-tune/val2017.zip -d data/validation2017_for_fine-tune
unzip -q data/validation2017_for_fine-tune/annotations_trainval2017.zip -d data/validation2017_for_fine-tune

#5) rename unziped data/train2017/train2017 dir -> to data/train2017/images,
# rename unziped data/validation2017_for_fine-tune/val2017 dir -> data/validation2017_for_fine-tune/images:
mv data/train2017/train2017 data/train2017/images
mv data/validation2017_for_fine-tune/val2017 data/validation2017_for_fine-tune/images

#6) move train2017 annotation files from validation2017 annotation dir
# to train2017 annotations dir:
mv data/validation2017_for_fine-tune/annotations/*train2017.json data/train2017/annotations

#7)One-Time step(for COLAB) - copy prepared data dir into drive/MyDrive(IF you have > 30GB free space):
!cp -r data drive/MyDrive/
!rm drive/MyDrive/data/train2017/*.zip
!rm drive/MyDrive/data/validation2017_for_fine-tune/*.zip

# ===IF USE COLAB ENV === #
# Cause of small amount of RAM memory, I will use Google Colab env for
# datasets creation for Stable Diffusion model training, validation. For this(as was tested),
# I use virtualenv as venv in order to install pycocotools(PythonAPI):
!pip3 install virtualenv
!virtualenv datasets_for_SD_model
!source datasets_for_SD_model/bin/activate
!pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# After that, we can start our dataset creation program. Make sure you choose right
# runtime GPU type and have enough disk, RAM memory.
# ========================= #

# Create project VENV(pwd - ../data) and libs for cocoAPI:
1)cd to project root dir: cd ../../../../
2)#1)create Project venv in project root dir - images_gen-tion_user_input-env:
  python3 -m venv "images_gen-tion_user_input-env"
  #2)activate venv: source images_gen-tion_user_input-env/bin/activate
  #3)pip install numpy==1.26
  #4)pip install scikit-image
  #5)pip install matplotlib
  # for GPU Runtime
  #6)pip install --upgrade setuptools wheel

# Installing COCO PythonAPI:
1)pwd - ../data_collection_and_preprocessing
2)pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
3)cp -r /home/kov_andrew/.local/lib/python3.10/site-packages/pycocotools .

# Installing Text-to-Image(Stable Diffusion) model modules, scripts for fine-tuning:
#1)a) git clone https://github.com/huggingface/diffusers
   b) pip install diffusers/
   c) pip install -r diffusers/examples/text_to_image/requirements.txt
#2)Init Accelerate environment(pwd - ../data_collection_and_preprocessing):
   a)accelerate config default

# Dir for saving of finetuned SD model(on model level):
mkdir finetuned_S_D_LoRA_model

# If training script has xFormers error, execute cmd:
pip install torch==2.7.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --force-reinstall --no-deps --pre xformers

# Now, time to model fine-tuning by executing bash script:
stable_diffusion_model_fine-tune_training.sh

# All details and additional steps
# see in Colab Notebook file - Stable_Diffusion_Model_Fine-Tuning_with_Dataset_Preparation.ipynb
