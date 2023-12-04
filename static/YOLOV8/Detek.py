from ultralytics import YOLO

# Membuat objek model YOLO dengan file bobot 'best.pt'
model = YOLO('best.pt')

# Mendeteksi objek dalam sumber gambar atau video
result = model(source=0, show=True, conf=0.3,save=True)