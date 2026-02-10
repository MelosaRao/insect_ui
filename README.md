## How to use this repo
### Installation
1. Make sure you install python version 3.11 (make sure you check the check box for saving the path variable though it also works on 3.12 on colab, we have observed issues with the newer 3.13 version)
2. Make sure you vscode is updated if vscode terminal throws errors you can run the same code in command prompt

### Setup Locally
1. Clone the github repo on your machine, navigate to the directory of your choice in command line run(this is with ssh): `git clone -b UI_Improved git@github.com:MelosaRao/insect_ui.git`
2. Navigate to the insect_ui directory that you just cloned and create and activate virtual environmennt (optional but recommended)
   - For windows to create: `python -m venv myenv` or for mac to create: `python3 -m venv myenv`
   - For windows to activate: `myenv\Scripts\activate.bat` or for mac to activate: `source myenv/bin/activate`
4. Run `pip install -r requirements.txt` and `pip install -r requirements_ml.txt` on command line. Make sure to resolve all dependencies before proceeding
5. Create a `models` folder inside insect_ui
6. Run `pip install gdown`
7. Run `gdown 14iCE6ps3WOSHPAmM5tzlrEX_IiROb4Zr -O cls_model.keras`,  `gdown 1IdQXwGsizccY9TSPiL2dMmVFUAZ58NRr -O detect_model.pt` on commandline to load model weights. Then save loaded model weights to `models` folder
8. Run `python run.py` to start app locally

### Usage
1. Run `python run.py` to start app locally
3. Click on upload page on top menu
4. Upload your image and click submit
5. Results will be displayed

### App Demo
App demo available at: https://drive.google.com/file/d/1wmeEnsF9dYi3I3yFRYkSEcPqwsU9ywL0/view?usp=sharing
