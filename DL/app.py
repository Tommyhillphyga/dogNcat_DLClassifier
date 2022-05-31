from torch.autograd import Variable
import numpy as np
import streamlit as st
import pandas as pd
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import PIL 
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader



def Transform_image(img_path):
    
    simple_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(img_path).convert('RGB')
    img_tr = simple_transform(img)
    img_t = img_tr.unsqueeze(0)
    return img_t

# img_path = './petImages/Dog/17.jpg'


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    device = torch.device('cpu')
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, 2)

    loaded_model = model
    loaded_model.load_state_dict(torch.load('./model.pth', map_location=device))
    loaded_model.eval()
    return loaded_model

st.title("Cat and Dog Classification")
 


def import_classify (image, model):
    
    with torch.no_grad():
        input = Variable(image)
        output = model(input)
        _,predicted = torch.max(output.data, 1)

    return predicted[0]

def main():

    model = load_model()
    file = st.file_uploader("please upload an image", type = ['jpg', 'png'])
    if file is None:
        st.text('please upload an image')
    else:
        image = Transform_image(file)
        display_img = Image.open(file)
        st.image(display_img, use_column_width=True)
        prediction = import_classify(image=image, model= model)

        if prediction ==  0:
            st.success('The image is a Cat')
        else:
            st.success('The image is a Dog')

    
    
if __name__=='__main__':
    main()

