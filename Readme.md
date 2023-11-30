CUEE sky image preprocessing;

1. Download sky images into `sky_images` folder 
2. Run `Step0_Extract_downloaded_image.py` to extract the zip files
3. Run `Step1_Sorting_files_to_dicts.py` to generate the h5 files by putting the files in input sequences and target output which is saved in h5 format. 
    - At this step, the image files will be converted to an array of size [1x12xHxW] where each sample is spaced by 1 second (you can manually adjust the spacing of 1 seconds to other number). 
    - Each h5 files will contains the input sequences and target output.
4. We have provided `Step2_dividing_training_testing.py` to arrange those h5 files where output is the text files that contain lists of images for training, testing and validation.
5. Lastly, `Step3_MakingTestingSets.py` will generate the h5 files for testing purpose:
    - It will use the list of testing images from previous steps.
    - It will save the images into h5 files with the original image size. 

- We have also provide the `dataloader_CUEE.py` to be used with Dataloader. 
 
