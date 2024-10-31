import numpy as np
import cv2 

def SAM(data: np.ndarray, signature: np.ndarray, threshold: float) -> np.ndarray:
    """
    Spectral Angle Mapper
    Args:
        data (np.ndarray): The hyperspectral data cube
        signature (np.ndarray): The spectral signature to compare
    Returns:
        np.ndarray: The mask with the pixels that are similar to the signature
    """
    spectral_angle = np.arccos(np.sum(signature * data, axis=2) / (np.linalg.norm(signature) * np.linalg.norm(data, axis=2)))
    threshold = threshold * (np.max(spectral_angle) - np.min(spectral_angle)) + np.min(spectral_angle)
    return spectral_angle, np.invert(spectral_angle < threshold)


def SC(data: np.ndarray, signature: np.ndarray, threshold: float) -> np.ndarray:
    """
    Spectral Correlation Index
    Args:
        data (np.ndarray): The hyperspectral data cube
        signature (np.ndarray): The spectral signature to compare
    Returns:
        np.ndarray: The mask with the pixels that are similar to the signature
    """
    spectral_correlation = np.sum(signature * data, axis=2) / (np.linalg.norm(signature) * np.linalg.norm(data, axis=2))
    # adjust the threshold to the range of the spectral correlation
    threshold = threshold * (np.max(spectral_correlation) - np.min(spectral_correlation)) + np.min(spectral_correlation)
    return spectral_correlation, np.array(spectral_correlation < threshold)



class HyperspectralAreaSelector:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.selected_area = None

    def _onclick(self, event, x, y, flags, param):
        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
            self.end_point = None

        # Update the rectangle's end point
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)

        # Finish drawing the rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
            self.selected_area = param[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0], :]
            print(f"Selected area from {self.start_point} to {self.end_point}")
            print(f"Shape of selected area: {self.selected_area.shape}")

    def __call__(self, data: np.ndarray):
        """
        Display an image and allow the user to select a rectangular area.
        """
        cv2.namedWindow("Select area and exit with q", cv2.WINDOW_NORMAL)  # Allow window to be resizable
        cv2.setMouseCallback("Select area and exit with q", self._onclick, param=data)

        while True:
            display_image = data.mean(axis=2).copy()
            if self.start_point and self.end_point:
                cv2.rectangle(display_image, self.start_point, self.end_point, (255, 0, 0), 1)

            cv2.imshow("Select area and exit with q", display_image)

            # Press 'q' to quit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.selected_area


class HyperspectralPixelSelector:
    def __init__(self):
        self.signature = None

    def _onclick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Selected pixel at (x, y): ({x}, {y})")
            self.signature = param[y, x, :]  # Access the spectral signature at the clicked pixel
            print(f"Spectral signature: {self.signature}")

    def __call__(self, data: np.ndarray):
        """
        Display an image and allow the user to select a pixel. 
        """
        cv2.namedWindow("Select pixel and exit with q", cv2.WINDOW_NORMAL)  # Allow window to be resizable
        cv2.setMouseCallback("Select pixel and exit with q", self._onclick, param=data)

        while True:
            cv2.imshow("Select pixel and exit with q", data.mean(axis=2))

            # Press 'q' to quit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.signature
    


class InteractiveHyperspectralImageProcessor:
    def __init__(self, data: np.ndarray, operation:callable, default_value=50, select_area=False):
        self.data = data
        self.operation = operation
        image1 = data.mean(axis=2)  # First image (interactive)

        self.end_point = None
        self.drawing = False
        self.select_area = select_area

        self.image1 = image1  # First image (interactive)
        self.image2 = np.zeros_like(image1)  # Second image (initially black)
        self.selected_pixel = None  # Store the selected pixel
        self.slider_value = default_value  # Initial slider value
        self.slider_value_float = default_value / 255  # Initial slider value in [0, 1]
        self.slider_active = False  # Slider is inactive until a pixel is clicked

        # Window names
        self.window1_name = "Select a pixel and adjust the threshold"
        self.window2_name = "Result Mask"

        # Create windows
        cv2.namedWindow(self.window1_name,cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window2_name,cv2.WINDOW_NORMAL)

        # Set mouse callback
        cv2.setMouseCallback(self.window1_name, self._on_mouse_click)

        # Create a trackbar (slider) in the first window
        cv2.createTrackbar("Threshold", self.window1_name, self.slider_value, 255, self._on_slider_change)

    def _on_mouse_click(self, event, x, y, flags, param):
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     self.selected_pixel = (x, y)
        #     self.slider_active = True  # Activate the slider
        #     print(f"Selected pixel: {self.selected_pixel}")
        
        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_pixel = (x, y)
            self.drawing = True
            self.slider_active = True  # Activate the slider
            self.end_point = None

        # Update the rectangle's end point
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.select_area:
            self.end_point = (x, y)

        # Finish drawing the rectangle
        elif event == cv2.EVENT_LBUTTONUP and self.select_area:
            self.end_point = (x, y)
            self.drawing = False
            self.selected_area = self.data[self.selected_pixel[1]:self.end_point[1], self.selected_pixel[0]:self.end_point[0], :]
            print(f"Selected area from {self.selected_pixel} to {self.end_point}")
            print(f"Shape of selected area: {self.selected_area.shape}")

    def _on_slider_change(self, value):
        if self.slider_active and self.selected_pixel:
            self.slider_value = value
            self.slider_value_float = value / 255
            self.update_result_image()

    def update_result_image(self):
        # Extract the pixel value from the selected point
        if self.select_area:
            pixel_value = self.selected_area.mean(axis=(0, 1))
        else:
            pixel_value = self.data[self.selected_pixel[1], self.selected_pixel[0],:]
        analysis,result = self.operation(self.data,pixel_value, self.slider_value_float)
        # Update the second image with the result
        #self.image2 = cv2.merge([result] * 3)  # Create a 3-channel image for display
        self.image2 = result
        binary = np.asarray(result, dtype="uint8")*255

        cv2.imshow(self.window2_name, binary)

    def run(self):
        while True:
            image1=self.image1.copy()
            if self.selected_pixel and self.end_point and self.select_area:
                cv2.rectangle(image1, self.selected_pixel, self.end_point, (255, 0, 0), 1)
            # Display the first image
            cv2.imshow(self.window1_name,image1)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up and close windows
        cv2.destroyAllWindows()
        return self.image2