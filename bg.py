import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def prepare_data():
    # Создаем структуру папок при необходимости
    base_dir = 'dataset'
    classes = ['you', 'not_you']
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    # Проверяем наличие изображений
    for cls in classes:
        if len(os.listdir(os.path.join(base_dir, cls))) == 0:
            raise ValueError(f"Папка {cls} пуста! Добавьте изображения и перезапустите программу.")

    # Настройка генераторов данных
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_gen, val_gen):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        verbose=1
    )
    model.save('face_recognition_model.keras')
    return model


def real_time_recognition(model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            resized = cv2.resize(face_roi, (150, 150))
            normalized = resized / 255.0
            prediction = model.predict(np.expand_dims(normalized, axis=0))[0][0]

            label = "YOU" if prediction > 0.75 else "UNKNOWN"
            color = (0, 255, 0) if label == "YOU" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Первичная настройка среды
    try:
        train_gen, val_gen = prepare_data()
    except Exception as e:
        print(e)
        exit()

    # Инициализация и обучение модели
    if not os.path.exists('face_recognition_model.keras'):
        model = create_model()
        model = train_model(model, train_gen, val_gen)
    else:
        model = load_model('face_recognition_model.keras')

    # Запуск распознавания
    real_time_recognition(model)