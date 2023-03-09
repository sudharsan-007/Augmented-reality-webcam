# Live Augmented Reality using April Tags (OpenCV)
By [Sudharsan Ananth](https://sudharsanananth.wixsite.com/sudharsan) 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-this-project">About this Project</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#run-the-code">How to run</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## About this Project 

This repo contains simple python script to create augmented reality using OpenCV and april tags. Camera calibration must be done in order to get stable AR results. For this purpose `capture_calibration_images.py` and `camera_calibration.py` is also also provided which captures images with a button press and produces calibration data with the images in a directory respectively. 


### Augmented Reality on Live Webcam

![Img_output_demo](assets/AR_demo3.gif)


## Dependencies 

This project is built with the below given major frameworks and libraries. The code is primarily based on python. 

* [Python](https://www.python.org/) 
* [NumPy](https://numpy.org)
* [OpenCV](https://docs.opencv.org/4.x/index.html) 

## Prerequisites

conda environment is ideal for creating environments with the right packages. Pip is required to install the dependencies.

* [Anaconda](https://www.anaconda.com) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [pip](https://pypi.org/project/pip/)


## Run the code

Simply clone the repo cd into the right directory and run the code. Step-by-Step instructions given below. 

1. Clone the repository using 
   ```sh
   git clone https://github.com/sudharsan-007/Augmented-reality-webcam.git
   ```

2. cd into the directory Augmented-reality-webcam
   ```sh
   cd Augmented-reality-webcam
   ```

3. Create a Environment using
   ```sh
   conda create -n ar_april_tag
   conda activate ar_april_tag
   ```

4. Install Dependencies
   ```sh
   pip install opencv-python
   ```


5. Run `capture_calibration_images.py` and capture some images for calibration.
    ```sh 
    python capture_calibration_images.py
    ```

6. Run `camera_calibration.py` to generate calibration matrix and distortion index.
    ```sh 
    python camera_calibration.py
    ```

7. Run `ar_april_tag.py` to generate calibration matrix and distortion index.
    ```sh 
    python ar_april_tag.py -t DICT_4X4_100
    ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>