import mediapipe as mp
import cv2

print(f"MediaPipe version: {mp.__version__}")

# Проверяем доступные модули
print("\nДоступные модули в mediapipe:")
print(dir(mp))

# Проверяем наличие solutions
if hasattr(mp, 'solutions'):
    print("\n✓ solutions найден")
    print("Доступные решения:", dir(mp.solutions))
else:
    print("\n✗ solutions не найден")