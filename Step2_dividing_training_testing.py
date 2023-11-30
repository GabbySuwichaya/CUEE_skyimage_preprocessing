import glob
import pdb
import os  
import h5py 
import numpy as np

def intersection(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))

def setunion(lst1, lst2, lst3):
    return sorted(list(set(lst1)) + list(set(lst2))  + list(set(lst3)) )

def exclude_from_list(lst_full, lst, total_index): 
    newlist = sorted(list(set(lst_full).difference(set(lst))))
    new_index = [total_index[index]  for index in range(len(lst_full)) if lst_full[index] not in lst] 
    excluded_index = [total_index[index]  for index in range(len(lst_full)) if lst_full[index] in lst] 
    return newlist, np.array(new_index), excluded_index

def read_text2list(filename):
    my_file = open(filename, "r") 
    data = my_file.read()
    data_into_list = data.split("\n") 
    my_file.close() 
    return data_into_list


def making_train_valid_test_list(root, imsize="512", percent_training=0.6, percent_valid=0.2, percent_testing=0.2):

    setting = "Tr%0.2f-Val%0.2f-Test%0.2f" % (percent_training, percent_valid, percent_testing)
    setting = setting.replace(".","p")

    h5_folder = "h5files_Frame-4-Mins_IMS-%s" % imsize
    h5_path   = os.path.join(root, h5_folder)

    all_h5_file_list = glob.glob("%s/*.h5" % h5_path)

    if len(all_h5_file_list) == 0: 
        if ((os.path.isfile("train_data_%s.txt" % setting)) or (os.path.isfile("test_data_%s.txt" % setting))) or (os.path.isfile("valid_data_%s.txt" % setting)) :

            list_training   = read_text2list("train_data_%s.txt" % setting)
            list_validating = read_text2list("test_data_%s.txt" % setting)
            list_testing    = read_text2list("valid_data_%s.txt" % setting)
            return list_training, list_validating, list_testing
        
        else:
            raise  ValueError('You have to find the way to list all the files because Text Files `test_/valid_/traing_....txt` does not exist. Or else you may need to generate h5files_Frame-4-Mins_IMS-/ folder by running Step 0 and Step 1')


    month_list      = ["%02d"   % ind   for ind   in range(3,11)]
    file_name_month_list = ["%s/%s-*-2023-**.h5" % ( h5_path,month) for month in month_list] 


    

    dict_index_testing_month     = {} 
    dict_index_valid_month       = {}
    dict_index_training_month    = {}

    dict_filelist_testing_month  = {}
    dict_filelist_valid_month    = {}
    dict_filelist_training_month = {}

    fix_testing_date = ["09-19-2023-10-13-00", "09-19-2023-10-40-00", "09-19-2023-10-56-00",  "09-19-2023-11-00-00", 
                        "09-19-2023-11-19-00", "09-19-2023-11-33-00", "09-19-2023-11-57-00",  "09-19-2023-12-20-00",

                        "09-22-2023-10-56-00", "09-22-2023-11-00-00", "09-22-2023-11-19-00",  "09-22-2023-11-33-00", 
                        "09-22-2023-11-57-00", "09-22-2023-12-04-00", "09-22-2023-12-14-00",  "09-22-2023-12-45-00",
                        "09-22-2023-12-57-00", "09-22-2023-13-02-00", "09-22-2023-13-22-00",  "09-22-2023-14-03-00", 
                        "09-22-2023-14-22-00", "09-22-2023-14-41-00", "09-22-2023-14-45-00", "09-22-2023-14-56-00",
                        "09-22-2023-15-35-00", "09-22-2023-15-46-00",
                        "09-30-2023-09-02-00", "09-30-2023-10-22-00", "09-30-2023-11-55-00",  "09-30-2023-12-52-00", 
                        "09-30-2023-13-12-00"]

    for month_, file_name_each_month in zip(month_list, file_name_month_list): 

        file_list_per_month = glob.glob(file_name_each_month)
        file_list_per_month.sort()
        num_file_per_month  =  len(file_list_per_month)

        if month_ == '09': 
            total_index = np.arange(num_file_per_month) 
            file_list_per_month_org = file_list_per_month
            file_list_per_month, total_index, excluded_index = exclude_from_list(file_list_per_month, fix_testing_date, total_index) 
            
        else:    
            file_list_per_month_org = file_list_per_month
            total_index = np.arange(num_file_per_month) 

        
        num_testing       = int(percent_testing*num_file_per_month)
        num_valid         = int(percent_valid*num_file_per_month)

        if month_ == '09':
            rand_index_testing     = total_index[np.random.randint(num_file_per_month, size=num_testing-len(fix_testing_date))]  
        else:
            rand_index_testing     = total_index[np.random.randint(num_file_per_month, size=num_testing)]  

        subtotal_1_index_list  = np.setdiff1d(total_index, rand_index_testing) 
        rand_index_valid       = subtotal_1_index_list[np.random.randint(len(subtotal_1_index_list), size=num_valid)] 

        training_index_list    = np.setdiff1d(subtotal_1_index_list, rand_index_valid)  

        

        if month_ == '09':
            dict_index_testing_month[month_]     = rand_index_testing.tolist() + excluded_index
        else:
            dict_index_testing_month[month_]     = rand_index_testing.tolist() 

        dict_index_valid_month[month_]       = rand_index_valid.tolist()
        dict_index_training_month[month_]    = training_index_list.tolist()


        assert intersection(dict_index_testing_month[month_], dict_index_valid_month[month_] ) == 0
        assert intersection(dict_index_testing_month[month_], dict_index_training_month[month_] ) ==  0
        assert intersection(dict_index_valid_month[month_],   dict_index_training_month[month_] ) == 0 
        assert intersection(setunion(dict_index_testing_month[month_],  dict_index_valid_month[month_],  dict_index_training_month[month_]), np.arange(len(file_list_per_month_org)).tolist()) == len(file_list_per_month_org) 


        if month_ == '09':
            dict_filelist_testing_month[month_]  = [ file_list_per_month[index] for index in rand_index_testing.tolist()] + fix_testing_date

        else:
            dict_filelist_testing_month[month_]  = [ file_list_per_month[index] for index in rand_index_testing.tolist()]

        dict_filelist_valid_month[month_]    = [ file_list_per_month[index] for index in rand_index_valid.tolist()]  
        dict_filelist_training_month[month_] = [ file_list_per_month[index] for index in training_index_list.tolist()]   

        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_valid_month[month_] ) == 0
        assert intersection(dict_filelist_testing_month[month_],  dict_filelist_training_month[month_] ) == 0
        assert intersection(dict_filelist_valid_month[month_],    dict_filelist_training_month[month_]) == 0 

        assert intersection(setunion(dict_filelist_testing_month[month_],   dict_filelist_valid_month[month_],  dict_filelist_training_month[month_]), file_list_per_month_org) == len(file_list_per_month_org) 

        print("[%s] Testing %d / Valid %d / Training %d" % (month_, len(dict_filelist_testing_month[month_]),len(dict_filelist_valid_month[month_]) ,len(dict_filelist_training_month[month_])))

    list_testing = []
    list_validating = []
    list_training = []

    print("======================== Put into List ==========================================")
    for month_, file_name_each_month in zip(month_list, file_name_month_list): 
        
        list_testing += dict_filelist_testing_month[month_]

        list_validating += dict_filelist_valid_month[month_]
        
        list_training += dict_filelist_training_month[month_]

        print("[%s] Testing %d / Valid %d / Training %d" % (month_, len(list_testing),len(list_validating) ,len(list_training)))


    with open("test_data_%s.txt" % setting, 'w') as output:
        for row in list_testing:
            row = os.path.basename(row)
            output.write(row + '\n')

    with open("valid_data_%s.txt" % setting, 'w') as output:
        for row in list_validating:
            row = os.path.basename(row)
            output.write(row + '\n')

    with open("train_data_%s.txt" % setting, 'w') as output:
        for row in list_training:
            row = os.path.basename(row)
            output.write(row + '\n')

    return list_training, list_validating, list_testing

if __name__ == "__main__":

    root      = os.getcwd()

    imsize = 512
    percent_training = 0.60
    percent_valid    = 0.20
    percent_testing  = 0.20

    making_train_valid_test_list(root, imsize, percent_training, percent_valid, percent_testing )