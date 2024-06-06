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
3. The ‘Input Path’ field corresponds to the path of the folder containing the worm images you want to analyze. You can either type the path or select the folder from the file explorer using the folder icon.
   
   <p align="center">
   <img width="400" alt="REDWorm GUI" src=/assets/input.PNG>
   </p>

4. Repeat the same process for the ‘Output Path’ field, which corresponds to the path where the results will be stored.
   
   <p align="center">
   <img width="400" alt="REDWorm GUI" src=/assets/output.PNG>
   </p>

Main results:
- **Obtain skeleton files**: the skeletons of the TIFF files located in the input folder are generated, where a TIFF file of the worm skeleton is obtained for each file in the folder.
- **CSV file**: a CSV file is obtained with the values of the parameters obtained in each frame of each TIFF file in the input folder. The parameters that make up this file are the following: ‘ImageName’, ‘Frame’, ‘Distance head-tail’, ‘Distance head-center’, ‘Distance centre-tail’, ‘Worm length’, ‘Coil (T/F)’, ‘Ratio head-tail’, ‘AVG Distance ABS’, ‘AVG Distance’, ‘Distances’, ‘Sign Classification’, ‘Max dist’, ‘Min dist’, ‘Max abs dist’, ‘Min abs dist’, ‘Skewness’, ‘Kurtosis’.
- **CSV file**: a CSV file is obtained with the average numerical parameters of each TIFF file conforming the input folder. Therefore, the average values of each file are placed in each line of the file. The average parameters stored in this file are: ‘ImageName’, ‘Distance head-tail’, ‘Distance head-center’, ‘Distance centre-tail’, ‘Worm length’, ‘Ratio head-tail’, ‘AVG Distance ABS’, ‘AVG Distance’, ‘Max dist’, ‘Min dist’, ‘Max abs dist’, ‘Min abs dist’, ‘Skewness’, ‘Kurtosis’.
- **Get graphics**: a folder is obtained with 3 graphics where the names of the files in the input folder are represented on the abscissa axis and the values of the parameters on the ordinate axis, represented by box plots. In each graph a different parameter is represented, these are the ‘Ratio head-tail’, ‘Distances’ and ‘Distances ABS’.

5. Choose the results you want to obtain by clicking on the options. The options will change their color to red when selected. You must choose at least one option and can choose up to four functionalities.
  
   <p align="center">
   <img width="400" alt="REDWorm GUI" src=/assets/seleccion.PNG>
   </p>

6. Click on the ‘Run’ button for running the app.

7. Wait for the operations to complete. A constantly moving progress bar will indicate that the program is running.
   
   <p align="center">
   <img width="400" alt="REDWorm GUI" src=/assets/progress_bar.PNG>
   </p>

8. When the progress bar stops, it indicates that the program has finished. A message will appear with the specified output path where the results are located.
   
   <p align="center">
   <img width="400" alt="REDWorm GUI" src=/assets/succed.PNG>
   </p>

9. To exit the application, simply close the application window.



