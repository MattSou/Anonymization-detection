import cv2
import numpy as np
from face_detection.yolov5_face.detector import Yolov5Face
import matplotlib.pyplot as plt
import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", 
                    type = str, 
                    help="choose the directory of the base dataset")
parser.add_argument('--write', action=argparse.BooleanOptionalAction)
parser.add_argument("--target_dir", 
                    type = str, 
                    help="target_directory for your synthetic dataset",
                    default=None)
args = parser.parse_args()

detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")


# print(args.base_dir)
# print(args.write)
# print(args.target_dir)

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None):
        self.img_labels = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        return img_path


class AnonymizePSF:
    def __init__(self, size=None):
        if size is None:
            size = np.random.randint(45,55)
        size = size//2 * 2 + 1
        self.size = size, size
        self.psf = self._create_psf()

    def _create_psf(self):
        x = np.linspace(-self.size[0] // 2, self.size[0] // 2, self.size[0])
        y = np.linspace(-self.size[1] // 2, self.size[1] // 2, self.size[1])
        x, y = np.meshgrid(x, y)
        psf = np.ones_like(x)
        return psf / np.sum(psf)

class GaussianPSF:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.psf = self._create_psf()

    def _create_psf(self):
        x = np.linspace(-self.size // 2, self.size // 2, self.size)
        y = np.linspace(-self.size // 2, self.size // 2, self.size)
        x, y = np.meshgrid(x, y)
        psf = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        return psf / np.sum(psf)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def imshow_cv2_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def find_face_and_blur_opt(img, show = False):
    bboxes, landmarks = detector.detect(image=img)
    face = {}
    laplacian = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    max_surface = 0
    h, w, c = img.shape
    #print(laplacian.shape)
    for i in range(len(bboxes)):
    # Get location of the face
        y1, x1, y2, x2, score = bboxes[i].astype('int')
        surface = (y2-y1)*(x2-x1)
        #print(x1,x2,y1,y2)
        lap = laplacian[x1:x2,y1:y2].var()
        blur_score = laplacian[x1:x2,y1:y2].var()*np.sqrt(h*w)*(y2-y1)*(x2-x1)/1e7
        d1 = np.linalg.norm(landmarks[i][1]-landmarks[i][0])
        d2 = np.linalg.norm(landmarks[i][3]-landmarks[i][4])
        front_score = np.mean([d1,d2])/np.sqrt(surface)
        if surface > max_surface:
            face={ 'coord':{
                'x1':x1,
                'y1':y1,
                'x2':x2,
                'y2':y2}
                , 
                'surface%':surface/(h*w),
                'laplacian': lap,
                'global_laplacian' : laplacian.var(),
                'blur_score':blur_score,
                'front_score':front_score
                }
            max_surface = surface        
    return face

def classify(dic, surface_threshold = 0.005, blur_threshold = 50, laplacian_threshold = 20, front_threshold = 0.1):
    if len(dic)==0:
        return 0
    surface = dic['surface%']>=surface_threshold
    blur = dic['blur_score']>=blur_threshold
    laplacian = dic['laplacian']>=laplacian_threshold
    front = dic['front_score']>=front_threshold
    if surface and blur and laplacian and front:
        return 1
    else: 
        return 0
    
def anonymize_opt(dic, img, show = False):
    
    x1, y1, x2, y2 = dic['coord']['x1'], dic['coord']['y1'], dic['coord']['x2'], dic['coord']['y2']
    sigma_y, sigma_x = (y2-y1)//2, (x2-x1)//2
    c = x1+sigma_x, y1+sigma_y

    # print(f"x1 : {x1}, x2 : {x2}, y1 : {y1}, y2 : {y2}")
    # print(f"center : {c}, sigma_x : {sigma_x}, sigma_y : {sigma_y}")

    blur_kernel = torch.tensor(AnonymizePSF().psf)
    kernel_size = blur_kernel.shape
    blur_kernel = blur_kernel.expand(3,1,blur_kernel.shape[0], blur_kernel.shape[1]).to(device='cuda').float()
    # print(blur_kernel.shape)
    smooth_kernel = torch.tensor(GaussianPSF(20,20).psf).unsqueeze(0).unsqueeze(0).to(device='cuda').float()

    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img_2 = torch.nn.functional.pad(img, padding, mode="reflect") 
    #print(blurred_image.shape)
    for i in range(3):
        blurred_image = torch.nn.functional.conv2d(img_2, blur_kernel, groups=3)


    x, y =np.mgrid[0:img.shape[-2], 0:img.shape[-1]]
    mask=((x-c[0])**2/(sigma_x+10)**2 + (y-c[1])**2/(sigma_y+10)**2 <= 1).astype('float')

    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device='cuda').float()
    mask = torch.nn.functional.conv2d(mask, smooth_kernel, padding = 'same')
    
    mask = (mask/torch.max(mask))
    # plt.imshow(cv2.rectangle(mask.cpu().numpy().squeeze(), (y1, x1), (y2, x2), (0, 146, 230), 2))
    # plt.colorbar()
    # plt.show()
    mask = mask.squeeze()
    mask = torch.stack([mask, mask, mask]).unsqueeze(0)


    blurred_image= img*(1-mask)+blurred_image*mask

    if show:
        b_i = blurred_image.squeeze().cpu().numpy().transpose(1,2,0).astype('uint8')
        a = cv2.rectangle(b_i, (y1, x1), (y2, x2), (0, 146, 230), 2)
        plt.imshow(a)
        plt.show()
    return blurred_image

def class_and_blur_opt(frame_path, target_path, write=False, show = False):
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = find_face_and_blur_opt(frame, show = show)
    face['label'] = classify(face)
    if face['label']==1:
        frame = torch.tensor(frame).permute(2,0,1).unsqueeze(0).to(device='cuda').float()
        anon = anonymize_opt(face, frame, show = show)
        if write:
            anon = anon.squeeze().cpu().numpy().transpose(1,2,0).astype('uint8')
            anon = cv2.cvtColor(anon, cv2.COLOR_RGB2BGR)
            cv2.imwrite(target_path, anon)
    else:
        anon = None
    return face, anon

def generate_synthetic_blurred_opt(base_path, loader, write = False, target_path=None, show = False):
    count = 0
    frames_info = {}
    if target_path is None:
        # target_path = os.path.join(os.path.dirname(base_path), base_path+'_anonymized')
        target_path = os.path.join(os.path.dirname(base_path), base_path+'_synth_anonymized')
        # print(target_path)
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
    json_path = os.path.join(target_path,'annotations.json')
    # for frame_path in tqdm(frame_list):
    for i, frame_path in enumerate(tqdm(loader)):
        frame_path = frame_path[0]
        # print(frame_path)
        anon_target_dir = '/'.join(frame_path.replace(base_path, target_path).split('/')[:-1])
        if not os.path.isdir(anon_target_dir):
            os.mkdir(anon_target_dir)
        # print(frame_path)
        #print(frame_path.replace(base_path, target_path))
        info, anon_frame = class_and_blur_opt(frame_path, frame_path.replace(base_path, target_path), write=write, show = show)
        frames_info[frame_path]=info
        count+=info['label']
    
    
    print("Number of anonymized frames: ", count)
            

    with open(json_path, 'w', encoding='utf-8') as f:
         json.dump(frames_info, f, ensure_ascii=False, indent=4, cls = NpEncoder)



def main():
    videos = []
    for file in os.listdir(args.base_dir):
        if not os.path.isfile(os.path.join(args.base_dir,file)):
            videos.append(file)
    frames = [os.listdir(os.path.join(args.base_dir,video)) for video in videos]

    paths = (
        pd.DataFrame({'videos': videos, 'frames': frames})
        .explode('frames')
        .dropna()
        .assign(img_path = lambda df : df.apply(lambda x : os.path.join(args.base_dir, x['videos'], x['frames']), axis=1))
        .drop(['videos', 'frames'], axis=1)
    )
    data = CustomImageDataset(dataframe=paths, transform=None)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=4)


    generate_synthetic_blurred_opt(args.base_dir, loader, write = args.write, target_path=args.target_dir, show = False)

if __name__=='__main__':
    main()