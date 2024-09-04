
This is a fork of the [Rope](https://github.com/Hillobar/Rope/) project (by Hillobar), which focus on implementing features for live swapping using Webcam and for live streaming the swapped videos using virtual camera.

For a full list of features and details, please visit the original project [page](https://github.com/Hillobar/Rope/)

## Additional Features in this Fork: ##

* Includes a lot of awesome mods created by Alucard24, Corza, Glat0s and others. You can check out Alucard's fork with those mods [here](https://github.com/Alucard24/Rope/)
* Supports using Live Webcam feed as a video source for swapping
* Supports sending the swapped video as Virtual Camera output (in realtime) which can be used in external applications like Google Meet, Zoom, Twitch etc..
* Integrates Rope with OBS Studio for real-time recording with microphone or computer audio
* Eyes and Mouth restoration from the original face (Blend the Eyes and Mouth from original face with the swapped face to preserve expressions)
* Supports using Deepfacelive models for swapping

## Requirements
  * Windows 10 or 11
  * Nvidia GPU with CUDA Support
  * [OBS Studio](https://obsproject.com/)

## Screenshots

![](https://i.ibb.co/fCLSjt0/Screenshot-609.png)

## Installation:

**Disclaimer**:
This repository is only intended for use on Windows (with Nvidia CUDA).

### Method 1: Using the Installer

1. Go to the [releases](https://github.com/argenspin/Rope-Live/releases/tag/1.0.0) page and download all the parts (_.bin files_) and the _Rope-Live Setup.exe_ for installation
2. Run the _Rope-Live Setup.exe_, choose a location to install on (Don't use the _Program Files_ folder or the root of the _C:_ drive) and complete the installation
3. Set up OBS Virtual Camera
   - Install and Start OBS Studio.
   - Click "Start Virtual Camera" (bottom right), then "Stop Virtual Camera".
   - Close OBS.
4. Start using the application by running the _Start_Rope.bat_ file from the installed directory

   #### Note: To get the latest updates, just run the _Update_Rope.bat_ file

### Method 2: Using Conda

#### Requirements
  * [Git for Windows](https://git-scm.com/download/win)
  * [Miniconda](https://docs.anaconda.com/miniconda/)
  * [FFMPEG](https://www.ffmpeg.org/download.html)


1. Open a folder in file explorer to install Rope in. Once you're in that folder, you can right-click it and select, 'Open in Terminal'

2. Clone the repository to your folder:

```
git clone https://github.com/argenspin/Rope-Live.git
cd Rope-Live
```

3. Create a new conda env and install required libraries

```
# create a conda venv with the correct python version
conda create -n Rope python=3.10.13

# activate the virtual environment
conda activate Rope

# install the dependencies
conda install nvidia/label/cuda-11.8.0::cuda-runtime cudnn=8.9.2.26 conda-forge::gputil=1.4.0
python -m pip install -r requirements.txt
```





4. Download the required models
  
   - **To get access to all the features of Rope**, you need to [download the models from here](https://github.com/argenspin/rope-assets/releases/tag/1.0.0). You need all of the files.
  
   - Place the downloaded model files in the `Rope-Live/models` folder

5. Set up OBS Virtual Camera
   - Start OBS.
   - Click "Start Virtual Camera" (bottom right), then "Stop Virtual Camera".
   - Close OBS.

7. Start the application by running ***Rope_Conda.bat*** file


 **Note: To use DFL models for swapping, put you models (.dfm) to the dfl_models directory**
### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me.
I've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
I would like to emphasize that our swapping software is intended for responsible and ethical use only. I must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. I encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.







  
