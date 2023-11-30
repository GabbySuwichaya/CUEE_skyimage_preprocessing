# %%
import glob
import pdb
import os
import imageio.v3 as iio
from tqdm import tqdm 
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from datetime import date, datetime, time

 

h5_image_dir = "sky_images_unzip"
all_images = glob.glob("%s/*/**/skycam_***.png" % h5_image_dir)
all_images.sort()

START_DATE =  date.fromisoformat('%04d-%02d-%02d' % (2023, 3, 15))
STOP_DATE  =  date.fromisoformat('%04d-%02d-%02d' % (2023, 11, 3))
data_frame = []

pbar = tqdm(all_images)
for image_path_name in pbar:

    image_name = os.path.basename(image_path_name)
    date_time  = image_name.split("skycam_")[1].split(".png")[0]
    year        = int(date_time[:4]) 
    month       = int(date_time[4:6])
    day         = int(date_time[6:8]) 
    hour        = int(date_time[8:10]) 
    minute      = int(date_time[10:12]) 
 

    DateTime_ISO    = '%04d-%02d-%02dT%02d:%02d:00' % (year, month, day, hour, minute) 
    
    DateTime = datetime.fromisoformat(DateTime_ISO) 
    Date     = date.fromisoformat('%04d-%02d-%02d' % (year, month, day))
    Time     = time.fromisoformat('%02d:%02d:00'   % (hour, minute)) 

    if Date >= START_DATE  and  Date <= STOP_DATE :

        time_minute = Time.hour * 60 + Time.minute

        data_frame.append({"Name": image_name, "DateTime": DateTime, "Date": Date, "Date_str": '%04d-%02d-%02d' % (year, month, day), "Minutes": time_minute,"Time":Time, "Year": year, "Month":month, "day":day, "hour":hour, "min":minute}) 
        pbar.set_description("%s" % DateTime)

data_frame = pd.DataFrame(data_frame)
data_frame.iloc[:10]

# %% [markdown]
# ### Select only Noon time

# %%
start_hour =  time.fromisoformat('%02d:%02d:00'   % (8, 00)) 
end_hour   =  time.fromisoformat('%02d:%02d:00'   % (16, 00)) 

selection = (data_frame["Time"] >= start_hour).values * (data_frame["Time"] <= end_hour).values 
data_frame_noon = data_frame.iloc[selection].copy() 

# %% [markdown]
# ### Stack images to BxCxHxW where C = 4 images x 3 RGB

# %%
import numpy as np

unique_days_list = list(np.unique(data_frame_noon["Date_str"]))  

FileName_list = []

for unique_day in unique_days_list:

    data_per_day = data_frame_noon.iloc[data_frame_noon["Date_str"].values == unique_day]
 

    for start_time in list(data_per_day["Minutes"].values): 

        diff_time_1 = 1
        diff_time_2 = 1
        diff_time_3 = 1
        diff_time_4 = 1

        time_0  =  start_time
        time_1  =  start_time + diff_time_1 
        time_2  =  start_time + diff_time_1  + diff_time_2 
        time_3  =  start_time + diff_time_1  + diff_time_2 + diff_time_3 
        time_4  =  start_time + diff_time_1  + diff_time_2 + diff_time_3  + diff_time_4 

        if (time_1 not in (data_per_day["Minutes"].values)) or (time_2 not in (data_per_day["Minutes"].values)) or (time_3 not in (data_per_day["Minutes"].values)) or (time_4 not in (data_per_day["Minutes"].values)):
            continue

        else:
             
            Date_Time = data_per_day.iloc[data_per_day["Minutes"].values == time_0].DateTime.values[0]         
            Name_0 = data_per_day.iloc[data_per_day["Minutes"].values == time_0].Name.values[0] 
            Name_1 = data_per_day.iloc[data_per_day["Minutes"].values == time_1].Name.values[0]
            Name_2 = data_per_day.iloc[data_per_day["Minutes"].values == time_2].Name.values[0]
            Name_3 = data_per_day.iloc[data_per_day["Minutes"].values == time_3].Name.values[0]
            Name_4 = data_per_day.iloc[data_per_day["Minutes"].values == time_4].Name.values[0]


            print("%s/%s/%s/%s/%s" %(Name_0,Name_1,Name_2, Name_3, Name_4))
            FileName_list.append({"Image_Files": [Name_0, Name_1, Name_2, Name_3], "GT_Files": Name_4, "Date":unique_day, "Date_Time":Date_Time}) 

FileName_list = pd.DataFrame(FileName_list)  
frame_duration = (diff_time_1 + diff_time_2 + diff_time_3 + diff_time_4)
# %% [markdown]
# ### Collect images and put them into arrays ....

# %%
from dataloader_CUEE import plot_patch
from PIL import Image
from Step2_dividing_training_testing import making_train_valid_test_list


im_resize = False
start_collecting_images = True 
if im_resize: 
    new_size = 480 #512
    Image_Size = "%d" % new_size
else:
    Image_Size = "%dx%d" % (1920,1080)

sub_folder_name = "h5files_Frame-%d-Mins_IMS-%s" % (int(frame_duration), Image_Size) 


root = os.getcwd() 
percent_training = 0.60
percent_valid    = 0.20
percent_testing  = 0.20  

_, _, list_testing =  making_train_valid_test_list(root, Image_Size, percent_training, percent_valid, percent_testing)
 

setting = "Tr%0.2f-Val%0.2f-Test%0.2f" % (percent_training, percent_valid, percent_testing)
folder_name_setting = "Testing-%s" % setting.replace(".","p") 
os.makedirs(folder_name_setting, exist_ok=True) 

folder_name = os.path.join(folder_name_setting, sub_folder_name)  
os.makedirs(folder_name, exist_ok=True)


if start_collecting_images:

    pbar_new = tqdm(FileName_list.iterrows())
    for index, row in pbar_new:

        Date_               = row["Date"].replace("-","_")
        input_image_names   = row["Image_Files"]
        gt_image_name       = row["GT_Files"]

        date_time_string       = row["Date_Time"].strftime("%m-%d-%Y-%H-%M-%S")
         
        if ("%s.h5" % date_time_string) not in list_testing:
            pbar_new.set_description("SKIP [%d/%d] %s.h5 Not for Testing" % (index, len(FileName_list), date_time_string))
        
        else:
            if os.path.isfile("%s/%s.h5" % (folder_name, date_time_string)): 
                pbar_new.set_description("[%d/%d] %s.h5 Exist: %s" % (index, len(FileName_list), date_time_string, folder_name))

            else:

                image_array_stack = []
                for image_path_name in input_image_names:
                    image_array = iio.imread("%s/%s/%s/%s" % (h5_image_dir, Date_,Date_, image_path_name)) 
                    

                    if im_resize: 
                        shape              = image_array.shape
                        nr, nc             = shape[0], shape[1]
                        img_pil            = Image.fromarray(image_array) 
                        shrinkFactor       = new_size /max([nr,nc])  
                        img_pil            = img_pil.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
                        image_array_pil    = np.array(img_pil) 

                        image_array_trans = np.transpose(image_array_pil,(2,1,0))  
                    
                    else:
                        # fig, axs = plt.subplots(1,2,figsize=(3, 3))  
                        # plot_patch(axs[0], image_array, "Predict")    
                        # plt.show()   

                        # pdb.set_trace() 
                        image_array_trans = np.transpose(image_array,(2,1,0)) 

                    image_array_stack.append(image_array_trans)

                image_array_stack = np.concatenate(image_array_stack, axis=0)    
                gt_image_array    = iio.imread("%s/%s/%s/%s" % (h5_image_dir, Date_, Date_, gt_image_name)) 

                if im_resize: 

                    shape               = gt_image_array.shape 
                    nr, nc              = shape[0], shape[1]
                    shrinkFactor        = new_size /max([nr,nc])  
                    img_pil_gt          = Image.fromarray(gt_image_array)
                    img_pil_gt          = img_pil_gt.resize((round(nc*shrinkFactor),round(nr*shrinkFactor)))
                    gt_image_array_pil  = np.array(img_pil_gt)
        
                    gt_image_array_trans = np.transpose(gt_image_array_pil,(2,1,0))  
                
                else:
                    gt_image_array_trans  = np.transpose(gt_image_array,(2,1,0)) 
                
                hf = h5py.File("%s/%s.h5" % (folder_name, date_time_string), 'w')
                hf.create_dataset('X', data=image_array_stack)
                hf.create_dataset('Y', data=gt_image_array_trans)
                hf.close() 
            
                pbar_new.set_description("[%d/%d] %s.h5 Folder: %s" % (index, len(FileName_list), date_time_string, folder_name))



