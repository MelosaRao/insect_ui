## How to use this repo

### Setup Locally
1. Clone the github repo on your machine and navigate to the insect_ui directory in command line
2. Create and activate virtual environmennt (optional)
3. Run `pip install -r requirements.txt` and `pip install -r requirements_ml.txt` on command line. Make sure to resolve all dependencies before proceeding
4. Create a `models` folder inside insect_ui
5. Run `pip install gdown`
6. Run `gdown 1eul2LTjjFX4ye3QXn2l4fNP6Cz1ltUIx -O cls_model.keras`,  `gdown 1IdQXwGsizccY9TSPiL2dMmVFUAZ58NRr -O detect_model.pt` on commandline to load model weights. Then save loaded model weights to `models` folder
7. Run `python run.py` to start app locally

### Usage
1. Run `python run.py` to start app locally
2. Create a new account and login
3. Click on upload page on top menu
4. Upload your image and click submit
5. Results will be displayed

### Sample Results video


https://github.com/user-attachments/assets/9b647816-d159-49a2-af6c-4af80d7aff75

