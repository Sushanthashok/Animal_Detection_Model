# image_predict.py
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

# Load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# All 95 class names (index must match your dataset)
class_names = ['Balaeniceps rex', 'Cybister', 'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'buffalo', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'giraffe', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybug', 'leopard', 'lion', 'lizard', 'lobster', 'mammoth', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'opossum', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

# Which animals are carnivores (red highlight)
carnivores = {'bear','cat','chimpanzee','dog','eagle','fox','gorilla','hyena','leopard','lion','opossum',
              'orangutan','owl','raccoon','rat','shark','snake','tiger','wolf'}

# ----------- PREDICT FUNCTION -----------
def detect_animals(image_path):
    results = model(image_path)[0]
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    carnivore_count = 0

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = int(cls.item())
        label = class_names[cls]

        # RED for carnivores, GREEN for others
        color = (255,0,0) if label in carnivores else (0,255,0)
        if label in carnivores:
            carnivore_count += 1

        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        draw.text((x1, y1-10), label, fill=color)

    output_path = "output_image.jpg"
    img.save(output_path)

    print(f"Output saved to {output_path}")
    print(f"Carnivores detected = {carnivore_count}")

# ----------- RUN THE FUNCTION -----------
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    detect_animals(image_path)
