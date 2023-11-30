1. Download sky_images_15-03-2023-to-15-04-2023/ folder  
2. Extract zip files ใน sky_images_15-03-2023-to-15-04-2023/ โดยการ run `extract_downloaded_image.py`
3. หลังจากได้ image file เรา จะ arrange มันเป็น array ขนาด [1x12xHxW] โดยสามารถ กำหนด ระยะห่างระหว่าง sample ได้ ตอนนี้ อาจารย์เซตไว้ ที่ 1 วินาที  และ save data ที่ถูกจัด  sorting_files_to_dicts.ipynb อาจารย์ save data ที่ถูกจัดใหม่ เป็น h5 file format
4. ให้ copy file ที่ชื่อ dataloader_CUEE.py ไปไว้ใน SkyImagers_ICCVW21/ folder ที่จะใช้ train และ copy pre_processing_images folder ไปไว้ใต้ folder เดียวกัน เช่น SkyImagers_ICCVW21/pre_processing_images ... แล้วให้ลอง run ไฟลล์ dataloader_CUEE.py ถ้ารันแล้วรูป ปรากฏ แสดงว่า ผ่าน
