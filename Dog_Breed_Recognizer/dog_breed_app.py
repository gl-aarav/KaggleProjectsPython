import cv2
from keras.models import load_model
import numpy as np
import streamlit as st

model = load_model('dog_breed_model.keras')