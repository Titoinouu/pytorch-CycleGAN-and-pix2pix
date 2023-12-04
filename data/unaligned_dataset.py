import os
from data.base_dataset import BaseDataset, get_transform, get_transform_gif
from data.image_folder import make_dataset
import random

import numpy as np
from PIL import Image, ImageSequence
import torch
import torchvision.transforms as transforms 

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        # MODIFICATIONS
        """
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        """
        
        self.transform_A = get_transform_gif(self.opt, grayscale=True)
        self.transform_B = get_transform_gif(self.opt, grayscale=True)
        
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.dummy_tensor = torch.zeros((self.load_size,self.load_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        
        input_nc = 20
        crop_size = 256
        
        
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_GIF = Image.open(A_path)
        B_GIF = Image.open(B_path)
        
        """ On veut appliquer les mêmes transformations aux frames d'un même gif"""        
        
        i, j, h, w = transforms.RandomCrop.get_params(
            self.dummy_tensor, output_size=(self.crop_size,self.crop_size))
        
        random_flip = False
        if random.random() > 0.5:
            random_flip = True
        
        A_stacked_list = []
        B_stacked_list = []
    
        
        """
        for frame_A,frame_B in zip(A_GIF,B_GIF):
            A_stacked_list.append(self.transform_A(frame_A))
            B_stacked_list.append(self.transform_B(frame_B))
        """   
        
        
        #print(f"\n Nombre de frame dans le gif A : {A_GIF.n_frames} \n")
        for index_A in range(A_GIF.n_frames):
            if index_A == 0 :
                frame_A = A_GIF
            if index_A > 0:
                A_GIF.seek(index_A)
                frame_A = A_GIF
            if frame_A is None:
                print(f"\n A_path : {A_path} \n n_frames : {A_GIF.n_frames} \n index_A : {index_A} \n")
                print(frame_A)
            else:
                frame_A = self.transform_A(frame_A)
                frame_A = transforms.functional.crop(frame_A,i,j,h,w)
                if random_flip:
                    frame_A = transforms.functional.hflip(frame_A)
                #print(f"frameA.shape : {frame_A.shape}")
                A_stacked_list.append(frame_A)
                
           
        i, j, h, w = transforms.RandomCrop.get_params(
            self.dummy_tensor, output_size=(self.crop_size,self.crop_size))
         
        random_flip = False
        if random.random() > 0.5:
            random_flip = True   
                
                
                
        for index_B in range(B_GIF.n_frames):
            if index_B == 0:
                frame_B = B_GIF
            if index_B > 0:
                B_GIF.seek(index_B)
                frame_B = B_GIF
            if frame_B is None:
                print(f"\n B_path : {B_path} \n n_frames : {B_GIF.n_frames} \n index_B : {index_B} \n")
                print(frame_B)
            else:
                frame_B = self.transform_B(frame_B)
                frame_B = transforms.functional.crop(frame_B,i,j,h,w)
                if random_flip:
                    frame_B = transforms.functional.hflip(frame_B)
                #print(f"frameB.shape : {frame_B.shape}")
                B_stacked_list.append(frame_B)
        
        if len(A_stacked_list) == 1:
            print(f"A_path : {A_path}, \n A_GIF.n_frames : {A_GIF.n_frames}")
        if len(B_stacked_list) == 1 :
            print(f"B_path : {B_path}, \n B_GIF.n_frames : {B_GIF.n_frames}")
        
        A_last_frame = A_stacked_list[-1]
        B_last_frame = B_stacked_list[-1]
         
        A_padding_to_do = input_nc - len(A_stacked_list)
        if A_padding_to_do < 0:
            raise ValueError(f"The value padding_to_do must be positive or null. \n path concerned : {A_path}")
        B_padding_to_do = input_nc - len(B_stacked_list) 
        if B_padding_to_do < 0:
            raise ValueError(f"The value padding_to_do must be positive or null. \n path concerned : {B_path}")
         
        for A_index in range(1,len(A_stacked_list)):
            A_stacked_list[0] = torch.cat((A_stacked_list[0],A_stacked_list[A_index]), axis = 0)
        
        if A_padding_to_do > 0:
            #print(f"A_stacked_list[0].shape avant : {A_stacked_list[0].shape}")
            repeated_tensor = A_last_frame.repeat((A_padding_to_do,) + (1,) * (A_last_frame.dim() - 1))
            #print(f"repeated_tensor.shape : {repeated_tensor.shape}")
            stacked_tensor = torch.cat([repeated_tensor], dim=0)
            #print(f"stacked_tensor.shape : {stacked_tensor.shape}")
            A_stacked_list[0] = torch.cat((A_stacked_list[0],stacked_tensor),0)
            #print(f"A_stacked_list[0].shape après : {A_stacked_list[0].shape}")
        
        for B_index in range(1,len(B_stacked_list)):
            B_stacked_list[0] = torch.cat((B_stacked_list[0],B_stacked_list[B_index]), axis = 0)
        
        if B_padding_to_do > 0:
            #print(f"A_stacked_list[0].shape avant : {A_stacked_list[0].shape}")
            repeated_tensor = B_last_frame.repeat((B_padding_to_do,) + (1,) * (B_last_frame.dim() - 1))
            #print(f"repeated_tensor.shape : {repeated_tensor.shape}")
            stacked_tensor = torch.cat([repeated_tensor], dim=0)
            #print(f"stacked_tensor.shape : {stacked_tensor.shape}")
            B_stacked_list[0] = torch.cat((B_stacked_list[0],stacked_tensor),0)
            #print(f"A_stacked_list[0].shape après : {A_stacked_list[0].shape}")
        """
        for B_index in range(1,len(B_stacked_list)) :
            B_stacked_list[0] = np.concatenate((B_stacked_list[0],B_stacked_list[B_index]), axis = 0)
        
        if B_padding_to_do > 0:
            print(B_stacked_list[0].shape, np.concatenate([B_last_frame[np.newaxis, :] for _ in range(B_padding_to_do)], axis=0).shape)
            B_stacked_list[0] = np.concatenate((B_stacked_list[0], np.concatenate([B_last_frame[np.newaxis, :] for _ in range(B_padding_to_do)], axis=0)), axis=0)
            
        """  
        """
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        """

        return {'A': A_stacked_list[0], 'B': B_stacked_list[0], 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
