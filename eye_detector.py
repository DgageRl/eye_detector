import cv2
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request
import threading  # для запуска видео в отдельном потоке

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ModernMediaPipeEyeDetector:
    def __init__(self):
        # ===== ПЕРЕМЕННЫЕ ДЛЯ ГЛАЗ =====
        # Индексы ключевых точек для глаз
        self.LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]

        # Индексы для вертикального измерения
        self.LEFT_EYE_VERTICAL = [159, 145]
        self.RIGHT_EYE_VERTICAL = [386, 374]

        # Индексы для горизонтального измерения
        self.LEFT_EYE_HORIZONTAL = [33, 133]
        self.RIGHT_EYE_HORIZONTAL = [362, 263]

        # ===== ПЕРЕМЕННЫЕ ДЛЯ СТАБИЛИЗАЦИИ =====
        self.ear_history = deque(maxlen=5)
        self.face_history = deque(maxlen=5)
        self.confirmation_threshold = 3

        # ===== ПЕРЕМЕННЫЕ СОСТОЯНИЯ =====
        self.eyes_closed_start = None
        self.alert_threshold = 2  # через сколько секунд включаем мем
        self.total_blinks = 0
        self.prev_eye_state = True
        self.last_state_change = 0
        self.state_change_delay = 0.2

        # ===== НОВЫЕ ПРЕМЕННЫЕ ДЛЯ ВИДЕО =====
        self.meme_thread = None  # поток с мемом
        self.meme_playing = False  # играет ли мем сейчас
        self.meme_window_name = "💩 МЕМНОЕ ВИДОСОВО 💩"  # название окна
        self.meme_video_path = "meme.mp4"  # путь к мемному видео (можно поменять)

        # ===== ПОРОГ ИЗ ВАШИХ ДАННЫХ =====
        self.ear_open = 0.17
        self.ear_closed = 0.13
        self.eye_threshold = (self.ear_open + self.ear_closed) / 2
        print(f"Порог установлен: {self.eye_threshold:.3f}")

        # ===== ДЛЯ РУССКИХ БУКВ =====
        self.setup_russian_font()

        # ===== ИНИЦИАЛИЗАЦИЯ MEDIAPIPE =====
        self.setup_mediapipe()

        print("\n" + "=" * 50)
        print("ДЕТЕКТОР ЗАПУЩЕН")
        print(f"MediaPipe version: {mp.__version__}")
        print("=" * 50)
        print(f"EAR открытых: {self.ear_open}")
        print(f"EAR закрытых: {self.ear_closed}")
        print(f"Порог: {self.eye_threshold}")
        print(f"Мем включится через: {self.alert_threshold} сек")
        print("=" * 50)

    def play_meme_video(self):
        """
        Проигрывает мемное видео в отдельном окне
        Запускается в отдельном потоке
        """
        # Проверяем существует ли файл с видео
        if not os.path.exists(self.meme_video_path):
            print(f"❌ Видео не найдено: {self.meme_video_path}")
            print("Положите файл meme.mp4 в папку с программой")
            return

        # Открываем видео
        cap = cv2.VideoCapture(self.meme_video_path)
        self.meme_playing = True

        # Получаем FPS видео для правильного воспроизведения
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)  # задержка между кадрами в миллисекундах

        print(f"🎬 МЕМ ЗАПУЩЕН! Видео: {self.meme_video_path}")

        while self.meme_playing:
            ret, frame = cap.read()
            if not ret:
                # Если видео закончилось, начинаем сначала
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Показываем видео в отдельном окне
            cv2.imshow(self.meme_window_name, frame)

            # Проверяем нажатие клавиш в окне с мемом
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC - закрыть мем
                break
            elif key == ord(' '):  # пробел - пауза
                cv2.waitKey(0)

        # Освобождаем ресурсы
        cap.release()
        cv2.destroyWindow(self.meme_window_name)
        self.meme_playing = False
        print("🎬 Мем закончился")

    def stop_meme(self):
        """Останавливает проигрывание мема"""
        if self.meme_playing:
            self.meme_playing = False
            if self.meme_thread and self.meme_thread.is_alive():
                self.meme_thread.join(timeout=1.0)

    def download_model(self):
        """Скачивает модель FaceLandmarker если её нет"""
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_path = "face_landmarker.task"

        if not os.path.exists(model_path):
            print("Скачивание модели FaceLandmarker...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("✓ Модель скачана")
            except Exception as e:
                print(f"✗ Ошибка скачивания: {e}")
                return None

        return model_path

    def setup_mediapipe(self):
        """Настройка MediaPipe"""
        try:
            model_path = self.download_model()
            if model_path is None:
                raise Exception("Не удалось получить модель")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("✓ MediaPipe инициализирован")

        except Exception as e:
            print(f"✗ Ошибка MediaPipe: {e}")
            print("Пробуем альтернативный метод...")
            self.setup_simple_mediapipe()

    def setup_simple_mediapipe(self):
        """Альтернативный метод с простым FaceMesh"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.use_simple_mp = True
            print("✓ Используется простой FaceMesh")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            raise

    def setup_russian_font(self):
        """Настройка шрифта для русских букв"""
        try:
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/Calibri.ttf",
            ]

            self.font = None
            for path in font_paths:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, 32)
                    self.font_small = ImageFont.truetype(path, 24)
                    self.font_big = ImageFont.truetype(path, 48)
                    break

            if self.font is None:
                self.font = ImageFont.load_default()

            self.use_pil = True
        except:
            self.use_pil = False

    def put_russian_text(self, img, text, position, size=32, color=(255, 255, 255)):
        """Рисует русский текст"""
        if not self.use_pil:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 20, color, 2)
            return img

        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            if size >= 48:
                font = self.font_big
            elif size >= 32:
                font = self.font
            else:
                font = self.font_small

            draw.text(position, text, font=font, fill=color[::-1])
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 20, color, 2)
            return img

    def calculate_ear(self, landmarks, vertical_idx, horizontal_idx, frame_shape):
        """Вычисляет Eye Aspect Ratio (EAR)"""
        h, w = frame_shape[:2]

        try:
            if hasattr(landmarks, 'landmark'):
                v1 = landmarks.landmark[vertical_idx[0]]
                v2 = landmarks.landmark[vertical_idx[1]]
                h1 = landmarks.landmark[horizontal_idx[0]]
                h2 = landmarks.landmark[horizontal_idx[1]]
            else:
                v1 = landmarks[vertical_idx[0]]
                v2 = landmarks[vertical_idx[1]]
                h1 = landmarks[horizontal_idx[0]]
                h2 = landmarks[horizontal_idx[1]]

            v1_point = np.array([v1.x * w, v1.y * h])
            v2_point = np.array([v2.x * w, v2.y * h])
            h1_point = np.array([h1.x * w, h1.y * h])
            h2_point = np.array([h2.x * w, h2.y * h])

            vertical_dist = np.linalg.norm(v1_point - v2_point)
            horizontal_dist = np.linalg.norm(h1_point - h2_point)

            ear = vertical_dist / (horizontal_dist + 1e-6)
            return ear

        except Exception as e:
            return 0.3

    def draw_eye_points(self, frame, landmarks, eye_indices, color):
        """Рисует точки вокруг глаз"""
        h, w = frame.shape[:2]

        try:
            for idx in eye_indices:
                if hasattr(landmarks, 'landmark'):
                    landmark = landmarks.landmark[idx]
                else:
                    landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, color, -1)
        except:
            pass

    def process_frame(self, frame):
        """Обработка кадра"""
        if hasattr(self, 'use_simple_mp') and self.use_simple_mp:
            return self.process_frame_simple(frame)
        else:
            return self.process_frame_new(frame)

    def process_frame_new(self, frame):
        """Обработка кадра через новый API"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_time_ms = int(time.time() * 1000)
            detection_result = self.detector.detect_for_video(mp_image, current_time_ms)

            face_detected = False
            ear_value = 0.3

            if detection_result.face_landmarks:
                face_detected = True
                landmarks = detection_result.face_landmarks[0]

                left_ear = self.calculate_ear(
                    landmarks,
                    self.LEFT_EYE_VERTICAL,
                    self.LEFT_EYE_HORIZONTAL,
                    frame.shape
                )

                right_ear = self.calculate_ear(
                    landmarks,
                    self.RIGHT_EYE_VERTICAL,
                    self.RIGHT_EYE_HORIZONTAL,
                    frame.shape
                )

                ear_value = (left_ear + right_ear) / 2.0

                self.draw_eye_points(frame, landmarks, self.LEFT_EYE_INDICES, (0, 255, 0))
                self.draw_eye_points(frame, landmarks, self.RIGHT_EYE_INDICES, (0, 255, 0))

            return frame, face_detected, ear_value
        except Exception as e:
            return frame, False, 0.3

    def process_frame_simple(self, frame):
        """Обработка кадра через простой FaceMesh"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            face_detected = False
            ear_value = 0.3

            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0]

                left_ear = self.calculate_ear(
                    landmarks,
                    self.LEFT_EYE_VERTICAL,
                    self.LEFT_EYE_HORIZONTAL,
                    frame.shape
                )

                right_ear = self.calculate_ear(
                    landmarks,
                    self.RIGHT_EYE_VERTICAL,
                    self.RIGHT_EYE_HORIZONTAL,
                    frame.shape
                )

                ear_value = (left_ear + right_ear) / 2.0

                self.draw_eye_points(frame, landmarks, self.LEFT_EYE_INDICES, (0, 255, 0))
                self.draw_eye_points(frame, landmarks, self.RIGHT_EYE_INDICES, (0, 255, 0))

            return frame, face_detected, ear_value
        except Exception as e:
            return frame, False, 0.3

    def run(self):
        """Основной цикл"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame = cv2.flip(frame, 1)

            # Обработка кадра
            frame, face_detected, ear_value = self.process_frame(frame)

            # Обновляем историю
            self.face_history.append(face_detected)
            self.ear_history.append(ear_value)

            # Сглаженный EAR
            smoothed_ear = sum(self.ear_history) / len(self.ear_history)

            # Стабильное состояние лица
            stable_face = sum(self.face_history) >= self.confirmation_threshold

            # Логика определения состояния
            if stable_face:
                eyes_closed = smoothed_ear < self.eye_threshold

                if current_time - self.last_state_change > self.state_change_delay:

                    if eyes_closed:
                        # ГЛАЗА ЗАКРЫТЫ
                        if self.eyes_closed_start is None:
                            self.eyes_closed_start = current_time
                            self.last_state_change = current_time

                            if self.prev_eye_state:
                                self.total_blinks += 1
                                print(f"👁 Морг! Всего: {self.total_blinks}")

                            self.prev_eye_state = False

                        closed_duration = current_time - self.eyes_closed_start

                        frame = self.put_russian_text(
                            frame,
                            f"ГЛАЗА ЗАКРЫТЫ! {closed_duration:.1f}с",
                            (50, 50), 36, (0, 0, 255)
                        )

                        # ===== ВКЛЮЧАЕМ МЕМ =====
                        if closed_duration > self.alert_threshold:
                            # Добавляем предупреждение на экран
                            frame = self.put_russian_text(
                                frame,
                                "🎬 СЕЙЧАС БУДЕТ МЕМ! 🎬",
                                (50, 100), 48, (0, 255, 255)
                            )

                            # Если мем еще не играет - запускаем
                            if not self.meme_playing:
                                print("🎬 ВКЛЮЧАЕМ МЕМ!!!")
                                self.meme_thread = threading.Thread(target=self.play_meme_video, daemon=True)
                                self.meme_thread.start()
                    else:
                        # ГЛАЗА ОТКРЫТЫ
                        if self.eyes_closed_start is not None:
                            self.last_state_change = current_time

                        self.eyes_closed_start = None
                        self.prev_eye_state = True

                        # Останавливаем мем если играет
                        if self.meme_playing:
                            print("👀 Глаза открыты, выключаем мем")
                            self.stop_meme()

                        frame = self.put_russian_text(
                            frame,
                            "ГЛАЗА ОТКРЫТЫ",
                            (50, 50), 36, (0, 255, 0)
                        )
            else:
                frame = self.put_russian_text(
                    frame,
                    "ЛИЦО НЕ ОБНАРУЖЕНО",
                    (50, 50), 32, (128, 128, 128)
                )
                # Если лицо пропало, выключаем мем
                if self.meme_playing:
                    self.stop_meme()

            # Отображение информации
            stats = [
                f"Морганий: {self.total_blinks}",
                f"EAR: {smoothed_ear:.3f}",
                f"Порог: {self.eye_threshold:.3f}",
                f"Состояние: {'ЗАКРЫТЫ' if smoothed_ear < self.eye_threshold else 'ОТКРЫТЫ'}",
                f"Мем: {'ИГРАЕТ' if self.meme_playing else 'ВЫКЛ'}"
            ]

            for i, stat in enumerate(stats):
                frame = self.put_russian_text(
                    frame, stat, (50, 150 + i * 30), 24, (255, 255, 255)
                )

            # График
            self.draw_ear_graph(frame, smoothed_ear, self.eye_threshold)

            # Подсказки
            frame = self.put_russian_text(
                frame, "ESC - выход | R - сброс", (50, 460), 20, (150, 150, 150)
            )

            cv2.imshow('Eye Detector', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r') or key == ord('к'):
                self.total_blinks = 0
                print("Счетчик сброшен")

        # Очистка при выходе
        self.stop_meme()
        cap.release()
        cv2.destroyAllWindows()

        print(f"\nВсего морганий: {self.total_blinks}")

    def draw_ear_graph(self, frame, current_ear, threshold):
        """Рисует график EAR"""
        graph_x = 50
        graph_y = 350
        graph_w = 400
        graph_h = 30

        cv2.rectangle(frame, (graph_x, graph_y),
                      (graph_x + graph_w, graph_y + graph_h),
                      (50, 50, 50), -1)
        cv2.rectangle(frame, (graph_x, graph_y),
                      (graph_x + graph_w, graph_y + graph_h),
                      (200, 200, 200), 1)

        bar_width = int((current_ear / 0.5) * graph_w)
        bar_width = min(bar_width, graph_w)

        color = (0, 255, 0) if current_ear > threshold else (0, 0, 255)
        cv2.rectangle(frame, (graph_x, graph_y),
                      (graph_x + bar_width, graph_y + graph_h),
                      color, -1)

        threshold_x = graph_x + int((threshold / 0.5) * graph_w)
        cv2.line(frame, (threshold_x, graph_y - 5),
                 (threshold_x, graph_y + graph_h + 5),
                 (255, 255, 0), 2)

        cv2.putText(frame, f"EAR: {current_ear:.3f}", (graph_x, graph_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == "__main__":
    detector = ModernMediaPipeEyeDetector()
    detector.run()