"""
FFT Frequency Domain Watermark Application
Built with DearPyGui for frequency domain watermark embedding
"""

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from pathlib import Path


class FFTWatermarkApp:
    def __init__(self):
        self.original_image = None
        self.result_image = None
        self.fft_channels = None
        self.image_path = ""
        
        # Watermark settings
        self.watermark_text = "WATERMARK"
        self.watermark_position = "center"  # left_top, right_top, left_bottom, right_bottom, center
        self.font_size = 32
        self.watermark_strength = 50.0
        self.fft_only_mode = False  # FFT only mode, no inverse transform
        self.spectrum_image = None  # Store spectrum visualization
        
        # Texture tags
        self.original_texture_tag = "original_texture"
        self.spectrum_texture_tag = "spectrum_texture"
        self.result_texture_tag = "result_texture"
        
        # Preview dimensions
        self.preview_width = 300
        self.preview_height = 300
        
    def setup_gui(self):
        """Create GUI interface"""
        dpg.create_context()
        
        # Register textures
        with dpg.texture_registry():
            # Create blank texture placeholders
            blank_data = [0.2] * (self.preview_width * self.preview_height * 4)
            dpg.add_dynamic_texture(
                width=self.preview_width, 
                height=self.preview_height, 
                default_value=blank_data,
                tag=self.original_texture_tag
            )
            dpg.add_dynamic_texture(
                width=self.preview_width, 
                height=self.preview_height, 
                default_value=blank_data,
                tag=self.spectrum_texture_tag
            )
            dpg.add_dynamic_texture(
                width=self.preview_width, 
                height=self.preview_height, 
                default_value=blank_data,
                tag=self.result_texture_tag
            )
        
        # Main window
        with dpg.window(label="FFT Frequency Domain Watermark", tag="main_window", width=1000, height=750):
            # Image input section
            dpg.add_text("=== Image Input ===")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="image_path_input",
                    hint="Enter image path...",
                    width=600,
                    callback=self.on_path_change
                )
                dpg.add_button(label="Browse", callback=self.show_file_dialog)
                dpg.add_button(label="Load Image", callback=self.load_image)
            
            dpg.add_separator()
            
            # Watermark settings section
            dpg.add_text("=== Watermark Settings ===")
            with dpg.group(horizontal=True):
                dpg.add_text("Watermark Text:")
                dpg.add_input_text(
                    tag="watermark_text_input",
                    default_value=self.watermark_text,
                    width=200,
                    callback=self.on_watermark_text_change
                )
            
            with dpg.group(horizontal=True):
                dpg.add_text("Position:")
                dpg.add_radio_button(
                    items=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"],
                    tag="position_radio",
                    default_value="Center",
                    horizontal=True,
                    callback=self.on_position_change
                )
            
            with dpg.group(horizontal=True):
                dpg.add_text("Font Size:")
                dpg.add_slider_int(
                    tag="font_size_slider",
                    default_value=self.font_size,
                    min_value=10,
                    max_value=100,
                    width=300,
                    callback=self.on_font_size_change
                )
            
            with dpg.group(horizontal=True):
                dpg.add_text("Strength:")
                dpg.add_slider_float(
                    tag="strength_slider",
                    default_value=self.watermark_strength,
                    min_value=1.0,
                    max_value=200.0,
                    width=300,
                    callback=self.on_strength_change
                )
            
            
            with dpg.group(horizontal=True):
                dpg.add_text("FFT Only Mode:")
                dpg.add_checkbox(
                    tag="fft_only_checkbox",
                    default_value=self.fft_only_mode,
                    callback=self.on_fft_only_change
                )
                dpg.add_text("(Skip inverse transform, view spectrum only)", color=(150, 150, 150))
            
            dpg.add_separator()
            
            # Image preview section
            dpg.add_text("=== Image Preview ===")
            with dpg.group(horizontal=True):
                with dpg.child_window(width=320, height=350):
                    dpg.add_text("Original Image")
                    dpg.add_image(self.original_texture_tag)
                
                with dpg.child_window(width=320, height=350):
                    dpg.add_text("Spectrum (with Watermark)")
                    dpg.add_image(self.spectrum_texture_tag)
                
                with dpg.child_window(width=320, height=350):
                    dpg.add_text("Result Image")
                    dpg.add_image(self.result_texture_tag)
            
            dpg.add_separator()
            
            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Process Image (Add Watermark)", 
                    callback=self.process_image,
                    width=200,
                    height=40
                )
                dpg.add_spacer(width=30)
                dpg.add_button(
                    label="Save Result", 
                    callback=self.save_result,
                    width=150,
                    height=40
                )
                dpg.add_spacer(width=30)
                dpg.add_button(
                    label="Save Spectrum", 
                    callback=self.save_spectrum,
                    width=150,
                    height=40
                )
            
            # Status bar
            dpg.add_separator()
            dpg.add_text("Status: Ready", tag="status_text")
        
        # File selection dialog
        with dpg.file_dialog(
            directory_selector=False, 
            show=False, 
            callback=self.file_dialog_callback,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))
            dpg.add_file_extension(".bmp", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")
        
        # Save file dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.save_dialog_callback,
            tag="save_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
        
        # Spectrum save file dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.spectrum_save_callback,
            tag="spectrum_save_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
        
        # Setup main window
        dpg.create_viewport(title='FFT Frequency Domain Watermark', width=1050, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def run(self):
        """Run the application"""
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def update_status(self, message):
        """Update status bar"""
        dpg.set_value("status_text", f"Status: {message}")
    
    def show_file_dialog(self):
        """Show file selection dialog"""
        dpg.show_item("file_dialog")
    
    def file_dialog_callback(self, sender, app_data):
        """File selection callback"""
        if app_data and "file_path_name" in app_data:
            file_path = app_data["file_path_name"]
            dpg.set_value("image_path_input", file_path)
            self.image_path = file_path
    
    def save_dialog_callback(self, sender, app_data):
        """Save file dialog callback"""
        if app_data and "file_path_name" in app_data:
            save_path = app_data["file_path_name"]
            if self.result_image is not None:
                # Ensure file extension
                if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    save_path += '.png'
                cv2.imwrite(save_path, self.result_image)
                self.update_status(f"Result saved to: {save_path}")
    
    def on_path_change(self, sender, app_data):
        """Path input change callback"""
        self.image_path = app_data
    
    def on_watermark_text_change(self, sender, app_data):
        """Watermark text change callback"""
        self.watermark_text = app_data
    
    def on_position_change(self, sender, app_data):
        """Position selection change callback"""
        position_map = {
            "Top-Left": "left_top",
            "Top-Right": "right_top", 
            "Bottom-Left": "left_bottom",
            "Bottom-Right": "right_bottom",
            "Center": "center"
        }
        self.watermark_position = position_map.get(app_data, "center")
    
    def on_font_size_change(self, sender, app_data):
        """Font size change callback"""
        self.font_size = app_data
    
    def on_strength_change(self, sender, app_data):
        """Watermark strength change callback"""
        self.watermark_strength = app_data
    
    def on_fft_only_change(self, sender, app_data):
        """FFT only mode change callback"""
        self.fft_only_mode = app_data
    
    def load_image(self):
        """Load image"""
        path = dpg.get_value("image_path_input")
        if not path:
            self.update_status("Please enter image path")
            return
        
        if not Path(path).exists():
            self.update_status(f"File not found: {path}")
            return
        
        try:
            # Load image with OpenCV (BGR format)
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                self.update_status("Failed to load image")
                return
            
            self.image_path = path
            self.update_texture(self.original_texture_tag, self.original_image)
            self.update_status(f"Loaded: {Path(path).name} ({self.original_image.shape[1]}x{self.original_image.shape[0]})")
            
        except Exception as e:
            self.update_status(f"Load error: {str(e)}")
    
    def update_texture(self, texture_tag, image):
        """Update texture display"""
        if image is None:
            return
        
        # Resize image to fit preview window
        h, w = image.shape[:2]
        scale = min(self.preview_width / w, self.preview_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create fixed size canvas
        canvas = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.uint8)
        
        # Center placement
        y_offset = (self.preview_height - new_h) // 2
        x_offset = (self.preview_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to RGBA and normalize
        rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
        data = rgba.flatten().astype(np.float32) / 255.0
        
        dpg.set_value(texture_tag, data.tolist())
    
    def apply_fft(self, image):
        """Apply FFT transform to RGB channels separately"""
        channels = cv2.split(image)
        fft_channels = []
        
        for c in channels:
            # Convert to float
            c_float = c.astype(np.float32)
            # FFT transform
            f = np.fft.fft2(c_float)
            # Shift spectrum center
            fshift = np.fft.fftshift(f)
            fft_channels.append(fshift)
        
        return fft_channels
    
    def create_watermark_mask(self, size, text, position, font_size):
        """Create watermark mask"""
        height, width = size
        
        # Create black background
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 30.0
        thickness = max(1, int(font_size / 15))
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Calculate text position
        margin = 20
        if position == "left_top":
            x = margin
            y = margin + text_height
        elif position == "right_top":
            x = width - text_width - margin
            y = margin + text_height
        elif position == "left_bottom":
            x = margin
            y = height - margin
        elif position == "right_bottom":
            x = width - text_width - margin
            y = height - margin
        else:  # center
            x = (width - text_width) // 2
            y = (height + text_height) // 2
        
        # Draw text
        cv2.putText(mask, text, (x, y), font, font_scale, 1.0, thickness)
        
        return mask
    
    def embed_watermark(self, fft_channels, watermark_mask, strength):
        """Embed watermark in frequency domain"""
        watermarked_channels = []
        
        for fshift in fft_channels:
            # Calculate average magnitude as reference
            magnitude = np.abs(fshift)
            avg_magnitude = np.mean(magnitude)
            
            # Embed watermark
            watermarked = fshift + strength * avg_magnitude * watermark_mask
            watermarked_channels.append(watermarked)
        
        return watermarked_channels
    
    def apply_ifft(self, fft_channels):
        """Apply inverse FFT transform to three channels"""
        channels = []
        
        for fshift in fft_channels:
            # Inverse shift spectrum
            f = np.fft.ifftshift(fshift)
            # Inverse FFT transform
            img = np.fft.ifft2(f)
            # Take absolute value of real part
            img = np.abs(img.real)
            # Clip and convert to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
            channels.append(img)
        
        return cv2.merge(channels)
    
    def create_spectrum_visualization(self, fft_channels):
        """Create spectrum visualization image"""
        spectrum_channels = []
        
        for fshift in fft_channels:
            # Calculate magnitude spectrum
            magnitude = np.abs(fshift)
            # Log transform for display
            magnitude_log = np.log(magnitude + 1)
            # Normalize to 0-255
            magnitude_normalized = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)
            spectrum_channels.append(magnitude_normalized)
        
        return cv2.merge(spectrum_channels)
    
    def process_image(self):
        """Process image - add frequency domain watermark"""
        if self.original_image is None:
            self.update_status("Please load an image first")
            return
        
        try:
            self.update_status("Processing...")
            
            # Get watermark text
            self.watermark_text = dpg.get_value("watermark_text_input")
            if not self.watermark_text:
                self.watermark_text = "WATERMARK"
            
            # 1. Apply FFT transform
            self.fft_channels = self.apply_fft(self.original_image)
            
            # 2. Create watermark mask
            h, w = self.original_image.shape[:2]
            watermark_mask = self.create_watermark_mask(
                (h, w), 
                self.watermark_text,
                self.watermark_position,
                self.font_size
            )
            
            # 3. Embed watermark in frequency domain
            watermarked_fft = self.embed_watermark(
                self.fft_channels, 
                watermark_mask, 
                self.watermark_strength
            )
            
            # 4. Create spectrum visualization
            spectrum_vis = self.create_spectrum_visualization(watermarked_fft)
            self.spectrum_image = spectrum_vis  # Store for saving
            self.update_texture(self.spectrum_texture_tag, spectrum_vis)
            
            # 5. Apply inverse FFT transform (if not FFT only mode)
            if not self.fft_only_mode:
                self.result_image = self.apply_ifft(watermarked_fft)
                self.update_texture(self.result_texture_tag, self.result_image)
                self.update_status("Done! Watermark embedded in frequency domain")
            else:
                self.result_image = None
                # Clear result preview
                blank_data = [0.2] * (self.preview_width * self.preview_height * 4)
                dpg.set_value(self.result_texture_tag, blank_data)
                self.update_status("FFT Done! Spectrum generated (FFT Only mode)")
            
        except Exception as e:
            self.update_status(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_result(self):
        """Save result image"""
        if self.result_image is None:
            self.update_status("No result to save, please process an image first (disable FFT Only mode)")
            return
        
        dpg.show_item("save_dialog")
    
    def save_spectrum(self):
        """Save spectrum image"""
        if self.spectrum_image is None:
            self.update_status("No spectrum to save, please process an image first")
            return
        
        dpg.show_item("spectrum_save_dialog")
    
    def spectrum_save_callback(self, sender, app_data):
        """Spectrum save dialog callback"""
        if app_data and "file_path_name" in app_data:
            save_path = app_data["file_path_name"]
            if self.spectrum_image is not None:
                # Ensure file extension
                if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    save_path += '.png'
                cv2.imwrite(save_path, self.spectrum_image)
                self.update_status(f"Spectrum saved to: {save_path}")


def main():
    app = FFTWatermarkApp()
    app.setup_gui()
    app.run()


if __name__ == "__main__":
    main()
