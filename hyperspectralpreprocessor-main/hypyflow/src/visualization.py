import cv2
import numpy as np
import matplotlib.pyplot as plt

class HyperspectralViewer:
    def __init__(self, cube, wavelengths=None):
        self.hsi = cube  # Hyperspectral image cube
        self.band_index = 0
        self.spectrum = None
        self.clicked_pixel_coords = None
        self.zoom_factor = 1.0
        self.equalize_band = False  # Flag to toggle band normalization
        self.show_tooltip_window = False  # Flag to show/hide tooltip window
        self.resize_ratio = 1.0
        
        self.name= 'HSI Band Viewer - "t" for tooltip'
        self.trackbar_equalize_name= 'Equalize (e)'
        self.trackbar_resize_name= 'Resize (r)'


        self.wavelengths = None
        self.provided_wavelegths = False
        if wavelengths is None:
            self.wavelengths = np.arange(self.hsi.shape[2])
        else:
            self.wavelengths = wavelengths
            self.provided_wavelegths = True

        # Create OpenCV window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

        # Create a slider for band selection
        cv2.createTrackbar('Band', self.name, 0, self.hsi.shape[2] - 1, self.update_band)
        cv2.createTrackbar(self.trackbar_equalize_name, self.name, 0, 1, self.toggle_equalize)
        cv2.setTrackbarPos(self.trackbar_equalize_name, self.name, int(self.equalize_band))
        cv2.createTrackbar(self.trackbar_resize_name, self.name, 0, 200, self.set_resize_ratio)
        cv2.setTrackbarMin(self.trackbar_resize_name, self.name, -50)

        # Set mouse callback for pixel selection
        cv2.setMouseCallback(self.name, self.mouse_callback)

        # Display the first band initially
        self.update_band(0)
    
    def set_resize_ratio(self, val):
        self.resize_ratio = 1 + val / 100
        self.show_band(self.hsi[:, :, self.band_index])

    def toggle_equalize(self, val):
        self.equalize_band = bool(val)
        self.update_band(self.band_index)

    # Callback function for the slider
    def update_band(self, val):
        self.band_index = val
        display_band = self.hsi[:, :, self.band_index]  # Get the band from HSI
        self.show_band(display_band)

    # Function to display the band with current settings (contrast, brightness, zoom, normalization)
    def show_band(self, band):
        
        if self.equalize_band:
            band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
            band = band.astype(np.uint8)  # Convert to uint8 for display
            band_processed = cv2.equalizeHist(band)
        else:
            band_processed = band.astype(np.float32)

        if self.resize_ratio != 1.0:
            band_processed = cv2.resize(band_processed, None, fx=self.resize_ratio, fy=self.resize_ratio)
            

        # Apply zoom if necessary
        if self.zoom_factor != 1.0:
            h, w = band_processed.shape[:2]
            center_x, center_y = w // 2, h // 2
            new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
            left = max(0, center_x - new_w // 2)
            right = min(w, center_x + new_w // 2)
            top = max(0, center_y - new_h // 2)
            bottom = min(h, center_y + new_h // 2)
            zoomed_image = band_processed[top:bottom, left:right]
            zoomed_image = cv2.resize(zoomed_image, (w, h))
            band_processed = zoomed_image
        
        # Draw the tooltip button
        self.draw_tooltip_button(band_processed)
        
        # Display image
        cv2.imshow(self.name, band_processed)

    # Draw tooltip button on the image
    def draw_tooltip_button(self, image):
        button_x, button_y, button_width, button_height = 10, 10, 150, 40
        button_color = (0, 255, 0) if not self.show_tooltip_window else (0, 0, 255)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)  # Convert to uint8 for display
        cv2.rectangle(image, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
        cv2.putText(image, 'Show Tooltip', (button_x + 10, button_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Mouse callback function to capture pixel click and display the spectrum
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
            # Check if the tooltip button is clicked
            if 10 <= x <= 160 and 10 <= y <= 50:
                self.toggle_tooltip()
            else:
                self.clicked_pixel_coords = (x, y)
                # Get the spectral signature of the clicked pixel
                self.spectrum = self.hsi[y, x, :]
                print(f"Clicked pixel: {self.clicked_pixel_coords}, Spectral signature: {self.spectrum}")

                # Plot the spectral signature
                plt.figure()
                plt.plot(range(self.hsi.shape[2]), self.spectrum)
                plt.title(f'Spectral Signature at Pixel {self.clicked_pixel_coords}')
                plt.xlabel('Band Index')
                plt.ylabel('Reflectance')
                plt.show()

    # Toggle tooltip visibility
    def toggle_tooltip(self):
        self.show_tooltip_window = not self.show_tooltip_window
        if self.show_tooltip_window:
            self.show_tooltip()

    # Function to save spectral signature to a .npy file
    def save_spectral_signature(self):
        if self.spectrum is not None and self.clicked_pixel_coords is not None:
            np.save(f"spectral_signature_{self.clicked_pixel_coords}.npy", self.spectrum)
            print(f"Spectral signature saved as spectral_signature_{self.clicked_pixel_coords}.npy")
        else:
            print("No pixel selected. Please click on a pixel to select it first.")

    # Function to select a region of interest (ROI) and plot average spectrum
    def select_area_and_plot_spectrum(self):
        display_band = self.hsi[:, :, self.band_index]

        if self.equalize_band:
            band = cv2.normalize(display_band, None, 0, 255, cv2.NORM_MINMAX)
            band = band.astype(np.uint8)  # Convert to uint8 for display
            band_processed = cv2.equalizeHist(band)
        else:
            band_processed = display_band.astype(np.float32)

        # Select ROI using OpenCV's built-in function
        roi = cv2.selectROI(self.name, band_processed)

        # Extract the ROI
        x1, y1, w, h = roi
        if w > 0 and h > 0:  # Ensure ROI is valid
            roi_pixels = self.hsi[y1:y1 + h, x1:x1 + w, :]

            # Calculate the average spectral signature for the ROI
            avg_spectrum = np.mean(np.mean(roi_pixels, axis=0), axis=0)
            std_spectrum = np.std(np.std(roi_pixels, axis=0), axis=0)

            # Plot the average spectral signature
            plt.figure()
            plt.plot(range(self.hsi.shape[2]), avg_spectrum)
            plt.fill_between(range(self.hsi.shape[2]), avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, alpha=0.3)
            plt.title(f'Average Spectral Signature in ROI {roi}')
            plt.xlabel('Band Index')
            plt.ylabel('Reflectance')
            plt.show()

    # Band math (ratio, difference, sum)
    def calculate_band_math(self, band1, band2, operation='ratio'):
        if operation == 'ratio':
            result = self.hsi[:, :, band1] / (self.hsi[:, :, band2] + 1e-6)  # Avoid division by zero
        elif operation == 'difference':
            result = self.hsi[:, :, band1] - self.hsi[:, :, band2]
        elif operation == 'sum':
            result = self.hsi[:, :, band1] + self.hsi[:, :, band2]
        elif operation == 'product':
            result = self.hsi[:, :, band1] * self.hsi[:, :, band2]
        elif operation == 'mean':
            result = (self.hsi[:, :, band1] + self.hsi[:, :, band2]) / 2
        else:
            print("Invalid operation. Please enter 'ratio', 'difference', 'sum', 'product', or 'mean'.")
            return
        result = np.clip(result, 0, 255)

        norm_display = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow(f'Band {operation.capitalize()}', norm_display.astype(np.uint8))

    # Function to save the currently displayed band
    def save_displayed_band(self):
        band = self.hsi[:, :, self.band_index]
        adjusted_band = cv2.convertScaleAbs(band, alpha=self.contrast / 50, beta=self.brightness - 100)
        cv2.imwrite(f'band_{self.band_index}.png', adjusted_band)
        print(f"Band {self.band_index} saved as band_{self.band_index}.png")

    # Zooming and panning
    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 4.0)
        self.show_band(self.hsi[:, :, self.band_index])

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 1.0)
        self.show_band(self.hsi[:, :, self.band_index])

    # Toggle normalization for band display
    def toggle_equalization(self):
        self.equalize_band = not self.equalize_band
        cv2.setTrackbarPos(self.trackbar_equalize_name, self.name, int(self.equalize_band))
        print(f"Band equalization is {'enabled' if self.equalize_band else 'disabled'}.")
        self.show_band(self.hsi[:, :, self.band_index])

    # Display the tooltip in a popup window
    def show_tooltip(self, *args):
        tooltip_text = """Controls:
        - Band Selection: Use the slider to change the band.
        - Click on a Pixel: Displays the spectral signature.
        - 'a': Select an area (ROI) and plot the average spectral signature.
        - 's': Save the spectral signature of the selected pixel (png).
        - 'S': Save the currently displayed band (png).
        - 'm': Perform band math (ratio, difference, sum).
        - '+': Zoom in.
        - '-': Zoom out.
        - 'e': Toggle equalization of the band for display.
        - 'i': Show information on the current band.
        - 't': Show this tooltip.
        - 'q': Quit the viewer.
        """
        # Create a new window for the tooltip
        cv2.namedWindow('Tooltip', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tooltip', 600, 300)

        # Create a blank image for the tooltip background
        tooltip_image = np.zeros((300, 600, 3), dtype=np.uint8)

        # Draw the text on the tooltip image
        y0, dy = 30, 20  # Start y position and line height
        for i, line in enumerate(tooltip_text.split('\n')):
            if line.strip():
                cv2.putText(tooltip_image, line.strip(), (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Show the tooltip window
        cv2.imshow('Tooltip', tooltip_image)

    def show_band_info(self):
        band = self.hsi[:, :, self.band_index]
        print(f"Band {self.band_index} Info:")
        print(f" - Range: {np.min(band)} - {np.max(band)}")
        print(f" - Mean: {np.mean(band)} - Standard Deviation: {np.std(band)}")
        print(f" - Shape: {band.shape}")
        print(f" - Data type: {band.dtype}")
        if self.provided_wavelegths:
            print(f" - Wavelength: {self.wavelengths[self.band_index]} nm")

    # Main loop to run the viewer
    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF

            # Save spectral signature ('s' key)
            if key == ord('s'):
                self.save_spectral_signature()

            # Save displayed band ('S' key)
            elif key == ord('S'):
                self.save_displayed_band()

            # Zoom in ('+' key)
            elif key == ord('+'):
                self.zoom_in()

            # Zoom out ('-' key)
            elif key == ord('-'):
                self.zoom_out()

            # Select area for average spectrum ('a' key)
            elif key == ord('a'):
                self.select_area_and_plot_spectrum()

            # Band math (enter two bands and operation)
            elif key == ord('m'):
                band1 = int(input('Enter first band: '))
                band2 = int(input('Enter second band: '))
                operation = input('Enter operation (ratio, difference, sum, mean, product): ')
                self.calculate_band_math(band1, band2, operation)

            # Toggle normalization ('n' key)
            elif key == ord('e'):
                self.toggle_equalization()

            # Show tooltip ('t' key)
            elif key == ord('t'):
                self.show_tooltip()

            # Show info on band ('i' key)
            elif key == ord('i'):
                self.show_band_info()

            # Quit program ('q' key)
            elif key == ord('q'):
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()

