import cv2
import numpy as np


class ChessboardCalibrator:
    def __init__(self):
        self.CHESSBOARD_SIZE = (6, 9)
        self.SQUARE_SIZE = 0.026
        self.objpoints = []
        self.imgpoints = []

        self.objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= self.SQUARE_SIZE

    def collect_calibration_images(self, camera, num_images=50):
        images_captured = 0

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, None)

            if ret:
                cv2.drawChessboardCorners(display_frame, self.CHESSBOARD_SIZE, corners, ret)

                if len(self.imgpoints) < num_images:
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners)
                    images_captured += 1
                    print(f"\rЗахвачено изображений: {images_captured}/{num_images}", end="")

            cv2.putText(display_frame, f"Захвачено: {images_captured}/{num_images}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f"Найдено углов: {len(corners) if ret else 0}/{self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1]}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if ret else (0, 0, 255), 2)

            cv2.imshow('Калибровка', display_frame)

            key = cv2.waitKey(100)
            if key == 27 or images_captured >= num_images:
                break

        cv2.destroyAllWindows()
        return images_captured == num_images

    def calibrate(self, image_size):
        if len(self.objpoints) < 5:
            raise ValueError("Необходимо как минимум 5 успешных захватов для калибровки")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)

        mean_error = 0
        print("\nВекторы поворота (rvecs) и переноса (tvecs) для каждого изображения:")
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

            print(f"\nИзображение {i + 1}:")
            print(f"rvec: {rvecs[i].reshape(1, 3)}")
            print(f"tvec: {tvecs[i].reshape(1, 3)}")

        print(f"\nСредняя ошибка перепроецирования: {mean_error / len(self.objpoints)} пикселей")
        return mtx, dist, rvecs, tvecs

    def save_calibration(self, mtx, dist, rvecs, tvecs, filename='camera_calibration.npz'):
        np.savez(filename, camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"Калибровка сохранена в {filename}")


def setup_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Не удалось открыть камеру")
    return camera


def test_calibration(camera, mtx, dist):
    print("\nТестирование калибровки...")
    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y + h, x:x + w]

        combined = np.hstack((frame, dst))
        cv2.putText(combined, "Оригинал | Исправленное", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Тест калибровки', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    try:
        print("Настройка камеры...")
        camera = setup_camera()
        calibrator = ChessboardCalibrator()

        print("\nИнструкции по калибровке:")
        print("1. Держите шахматную доску под разными углами")
        print("2. Вся доска должна быть видна")
        print("3. Двигайте медленно, чтобы избежать размытия")
        print("4. Изображения будут захватываться автоматически")
        print("5. Нажмите ESC для выхода\n")

        input("Нажмите Enter для начала калибровки...")

        _, frame = camera.read()
        image_size = frame.shape[:2][::-1]

        success = calibrator.collect_calibration_images(camera)

        if success:
            print("\nВыполняется калибровка...")
            camera_matrix, dist_coeffs, rvecs, tvecs = calibrator.calibrate(image_size)
            calibrator.save_calibration(camera_matrix, dist_coeffs, rvecs, tvecs)

            print("\nМатрица камеры:")
            print(camera_matrix)
            print("\nКоэффициенты искажения:")
            print(dist_coeffs)

            test_calibration(camera, camera_matrix, dist_coeffs)

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()