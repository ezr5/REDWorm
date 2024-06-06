<p align="center">
  <img width="300" src=/assets/Redworm.png alt="logo">
  <h1 align="center" style="margin: 0 auto 0 auto;"></h1>
  <h4 align="center" style="margin: 0 auto 0 auto;"><em>C.elegans</em> Behaviour Analyser</h4>

## 💡 Introduction
REDWorm is a tool developed in Python that facilitates the analysis of _C.elegans_ images for scientific studies through a GUI. It works with file folders, i.e. the input of the program is always a folder. This can directly contain different TIFF files of the segmented _C.elegans_ or be made up of subfolders. In this way, the application automates several processes, including obtaining skeletons, generating CSV files with relevant data and creating graphs to visualise the results. 
<p align="center">
<img width="500" alt="REDWorm GUI" src=/assets/interfaz.PNG>
</p>

## 📢 Requirements
- The files to be analysed must be in TIFF format.
- These files have to correspond to the C.elegans segmentation obtained from the in vivo microscopy images.
- The files have to be stored in a folder.

## 📐 How it Works
Follow these steps to use the RedWorm software for analyzing worm images:
1. Download the executable file ‘RedWorm.exe’.
2. Open the file on your computer and wait until the GUI appears.
3. The ‘Input Path’ field corresponds to the path of the folder containing the worm images you want to analyze. You can either type the path or select the folder from the file explorer using the folder icon. <p align="center">
<img width="400" alt="REDWorm GUI" src=/assets/interfaz.PNG>
</p>
4. Repeat the same process for the ‘Output Path’ field, which corresponds to the path where the results will be stored.
5. Choose the results you want to obtain by clicking on the options. The options will change their color to red when selected. You must choose at least one option and can choose up to four functionalities.
6. Click on the ‘Run’ button.
7. Wait for the operations to complete. A constantly moving progress bar will indicate that the program is running.
8. When the progress bar stops, it indicates that the program has finished. A message will appear with the specified output path where the results are located.
9. To exit the application, simply close the application window.



