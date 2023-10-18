#this file opens every .zip file in the directory and shows the png files inside

import os
import zipfile
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="directory",)
args = parser.parse_args()
directory=args.directory


#iterate through all files in directory
for filename in os.listdir(directory):
    #if the file is a zip file
    if filename.endswith(".zip"):
       #check for bad magic number
        print(filename)
        with open(os.path.join(directory, filename), 'rb') as f:
            data = f.read()
            if data[0:2] != b'PK':
                print('Bad magic number for file %s' % filename)
            else:
                #try to open the zip file
                data

        try:
            
            with zipfile.ZipFile(os.path.join(directory, filename),"r") as z:

                #get the names of all files in the zip
                for name in z.namelist():
                    #print the name of the file

                    #if the file is a png file
                    if name[-4:] == ".png":
                        print(name)

                        #open the file
                        with z.open(name) as f:
                            #read the file into a buffer
                            b=io.BytesIO(f.read())
                            #open the buffer as an image
                            img = plt.imread(b)
                            #display the image
                            plt.imshow(img)
                            plt.show()
                            #close the buffer
                            b.close()
        except zipfile.BadZipFile:
            # Open the ZIP file in binary mode
            z=zipfile.ZipFile(os.path.join(directory, filename), 'r')      
            if z.testzip() is not None:
                # The ZIP file is corrupted
                print('Bad Zip file')

                # Fix the ZIP file
                z.extractall('fixed_zip')

                # Reopen the fixed ZIP file
                zip_file = zipfile.ZipFile('fixed_zip/{}'.format(filename), 'r')

                # Verify the fixed ZIP file
                if zip_file.testzip() is not None:
                    print('Failed to fix the Zip file')
                else:
                    print('Zip file fixed')
            else:
                # The ZIP file is valid
                print('Good Zip file')
                #get the names of all files in the zip
                for name in z.namelist():
                    #print the name of the file
                    print("recovered: {}".format(name))

                    #if the file is a png file
                    if name.endswith(".png"):
                        #open the file
                        with z.open(name) as f:
                            #read the file into a buffer
                            b=io.BytesIO(f.read())
                            #open the buffer as an image
                            img = plt.imread(b)
                            #display the image
                            plt.imshow(img)
                            plt.show()
                            #close the buffer
                            b.close()
            z.close()