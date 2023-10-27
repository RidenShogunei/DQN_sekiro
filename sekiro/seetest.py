
import cv2
import torch
import grabscreen


def preprocess_and_normalize_image(window_size):
    while True:
        # 获取窗口截图
        screenshot = grabscreen.grab_screen(window_size)

        # 将图像转换为灰度图
        #gray_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # 边缘突出处理
        edges = cv2.Canny(screenshot, threshold1=100, threshold2=200)

        # 归一化处理
        normalized_image = torch.from_numpy(edges).float() / 255.0

        # 显示原始图像和经过处理的图像
        cv2.imshow("Raw Image", screenshot)
        cv2.imshow("Processed Image", edges)

        # 按下 "q" 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return normalized_image

window_size = (700, 100, 1204, 1050)
preprocessed_image = preprocess_and_normalize_image(window_size)