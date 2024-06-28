import torch
import numpy as np
import torch_directml
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import deque

def load_model(model_path, num_classes):

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    device = torch_directml.device()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  
    return model, device

def preprocess_image(image):
 
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    image_tensor = transform(image)
    return image_tensor

def predict_image(model, device, image_tensor):

    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image_tensor)
        softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
        max_score, predicted = torch.max(softmax_scores, 1)

    return predicted.item(), max_score.item()

def draw_predicted_class(image, predicted_class, confidence):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 21)
    text = f'Predicted class: {predicted_class} ({confidence:.2f})'
    draw.text((50, 50), text, fill=(170, 51, 106), font=font)

    return image

if __name__ == '__main__':

    model_path = r  # change path to where model is saved in your pc
    num_classes = 27 
    class_names = [
         'dry_asphalt_severe', 'dry_asphalt_slight', 'dry_asphalt_smooth', 'dry_concrete_severe', 'dry_concrete_slight', 'dry_concrete_smooth', 'dry_gravel', 'dry_mud', 'fresh_snow', 'ice', 'melted_snow', 'water_asphalt_severe',
        'water_asphalt_slight', 'water_asphalt_smooth', 'water_concrete_severe', 'water_concrete_slight', 'water_concrete_smooth', 'water_gravel', 'water_mud', 'wet_asphalt_severe', 'wet_asphalt_slight', 'wet_asphalt_smooth', 'wet_concrete_severe', 'wet_concrete_slight', 'wet_concrete_smooth', 'wet_gravel', 'wet_mud'
    ]
    model, device = load_model(model_path, num_classes)
    video_path = r #change path to where your test video is
    cap = cv2.VideoCapture(video_path)

    smoothing_window_size = 10
    predictions_deque = deque(maxlen=smoothing_window_size)
    confidence_threshold = 0.6  

    frame_count = 0
    save_frame_interval = 30 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        image_tensor = preprocess_image(frame_pil)
        
        predicted_index, confidence = predict_image(model, device, image_tensor)
        if confidence > confidence_threshold:
            predictions_deque.append(predicted_index)

        if predictions_deque:
            smoothed_prediction = np.bincount(predictions_deque).argmax()
            predicted_class = class_names[smoothed_prediction]
        else:
            predicted_class = "Uncertain"

        image_with_label = draw_predicted_class(frame_pil, predicted_class, confidence)
        frame_with_label = cv2.cvtColor(np.array(image_with_label), cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', frame_with_label)

        if frame_count % save_frame_interval == 0:
            frame_save_path = f"frame_{frame_count}.jpg"
            frame_pil.save(frame_save_path)
            print(f"Saved frame: {frame_save_path}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


