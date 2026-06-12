## How to use this repo
### Installation
1. Make sure you install python version 3.11 (make sure you check the check box for saving the path variable though it also works on 3.12 on colab, we have observed issues with the newer 3.13 version)
2. Make sure your vscode is updated if vscode terminal throws errors you can run the same code in command prompt

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
1. Start the app locally:

   - (Optional) Activate your virtual environment on Windows:

     ```powershell
     myenv\Scripts\activate.bat
     ```

   - Run the server:

     ```bash
     python run.py
     ```

2. Open your browser to http://127.0.0.1:5000 (or the address shown in the console).

3. Upload a sticky trap image via the "Upload" page and submit. Optional fill in text fields.

4. After processing you will see:
   - Annotated image with detection boxes
   - Detected class counts
   - Download links for Summary CSV, Detailed CSV, and COCO JSON

5. Review and correct detections using the Inline Editor:
   - Use the category filter or the select menu to pick a crop.
   - Change the class using the "Change to" dropdown and click "Save".
   - A special option, "Not an Insect", prompts for confirmation. If you confirm, that choice is final for the current session and the crop is removed from the review list.
   - Saving produces edited sidecar files (detailed CSV, summary CSV, COCO JSON) and makes download links available in the UI.
   - Addtional Information for annotators: The Diperans are threshold at 0.4 confidence. If the model's intial prediction was Diperan and it was filtered to other becuase of low confidence, that is the intended behavior. Feel free to edit and make changes.

6. Upload edited results to Roboflow using the "Upload Edited to Roboflow" button. A confirmation dialog appears because this overwrites the dataset entries on Roboflow. This updates your annotation to the dataset for future training!

### App Demo

App demo available at: https://drive.google.com/file/d/1wmeEnsF9dYi3I3yFRYkSEcPqwsU9ywL0/view?usp=sharing
