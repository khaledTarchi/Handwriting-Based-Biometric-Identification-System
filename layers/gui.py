"""
GUI Layer
=========
Layer 6 of the HB-BIS system

This layer implements the graphical user interface with three main modes:
1. Identification Mode: Upload handwriting sample for identification
2. Enrollment Mode: Register new user with handwriting samples
3. Management Mode: View stats and retrain models

A good biometric UI should:
- Provide clear feedback (match/uncertain/unknown)
- Show confidence scores (transparency)
- Guide users (tooltips, help text)
- Be accessible (large buttons, clear colors)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from typing import List, Optional
import threading

# Project imports
from config import *
from layers.data_acquisition import load_image
from layers.preprocessing import preprocess_for_svm, preprocess_for_squeezenet
from layers.feature_engineering import extract_svm_features, extract_squeezenet_features
from layers.database import enroll_user, get_all_users, get_database_stats
from layers.decision import identify_user, check_similarity_warning
from utils import validate_user_name
from models.squeezenet_model import retrain_squeezenet


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HBBIS_GUI:
    """
    Main GUI application for Handwriting-Based Biometric Identification System.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("HB-BIS - Handwriting-Based Biometric Identification System")
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=COLOR_BG)
        
        # State variables
        self.selected_model = tk.StringVar(value="squeezenet")
        self.current_images = []  # For enrollment
        self.current_image_path = None  # For identification
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create all UI widgets."""
        
        # ===== Header =====
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="HB-BIS",
            font=("Arial", 28, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Handwriting-Based Biometric Identification System",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle_label.pack(side=tk.LEFT, padx=10)
        
        # ===== Model Selection =====
        model_frame = tk.Frame(self.root, bg=COLOR_BG)
        model_frame.pack(pady=10)
        
        tk.Label(
            model_frame,
            text="Model Selection:",
            font=("Arial", 12, "bold"),
            bg=COLOR_BG
        ).pack(side=tk.LEFT, padx=10)
        
        svm_radio = tk.Radiobutton(
            model_frame,
            text="SVM (Classical - 42D Features)",
            variable=self.selected_model,
            value="svm",
            font=("Arial", 11),
            bg=COLOR_BG
        )
        svm_radio.pack(side=tk.LEFT, padx=10)
        
        squeezenet_radio = tk.Radiobutton(
            model_frame,
            text="SqueezeNet (Deep Learning - 512D Features)",
            variable=self.selected_model,
            value="squeezenet",
            font=("Arial", 11),
            bg=COLOR_BG
        )
        squeezenet_radio.pack(side=tk.LEFT, padx=10)
        
        # Info button
        info_btn = tk.Button(
            model_frame,
            text="ℹ Model Info",
            command=self.show_model_info,
            bg=COLOR_PRIMARY,
            fg="white",
            font=("Arial", 10)
        )
        info_btn.pack(side=tk.LEFT, padx=20)
        
        # ===== Main Content Area =====
        content_frame = tk.Frame(self.root, bg=COLOR_BG)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side: Identification & Enrollment
        left_frame = tk.Frame(content_frame, bg=COLOR_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Identification section
        self.create_identification_section(left_frame)
        
        # Enrollment section
        self.create_enrollment_section(left_frame)
        
        # Right side: Management
        right_frame = tk.Frame(content_frame, bg=COLOR_BG, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        right_frame.pack_propagate(False)
        
        self.create_management_section(right_frame)
    
    def create_identification_section(self, parent):
        """Create identification UI section."""
        frame = tk.LabelFrame(
            parent,
            text="🔍 Identification Mode",
            font=("Arial", 14, "bold"),
            bg="white",
            padx=15,
            pady=15
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Upload button
        upload_btn = tk.Button(
            frame,
            text="📁 Upload Handwriting Image",
            command=self.upload_for_identification,
            bg=COLOR_PRIMARY,
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            cursor="hand2"
        )
        upload_btn.pack(fill=tk.X, pady=5)
        
        # Image preview
        self.id_image_label = tk.Label(frame, text="No image loaded", bg="white", fg="gray")
        self.id_image_label.pack(pady=10)
        
        # Identify button
        self.identify_btn = tk.Button(
            frame,
            text="Identify User",
            command=self.perform_identification,
            bg=COLOR_MATCH,
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            state=tk.DISABLED
        )
        self.identify_btn.pack(fill=tk.X, pady=5)
        
        # Result display
        self.result_frame = tk.Frame(frame, bg="white")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_enrollment_section(self, parent):
        """Create enrollment UI section."""
        frame = tk.LabelFrame(
            parent,
            text="👤 Enrollment Mode",
            font=("Arial", 14, "bold"),
            bg="white",
            padx=15,
            pady=15
        )
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Name input
        name_frame = tk.Frame(frame, bg="white")
        name_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(name_frame, text="User Name:", font=("Arial", 11), bg="white").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_frame, font=("Arial", 11), width=25)
        self.name_entry.pack(side=tk.LEFT, padx=10)
        
        # Upload images button
        upload_btn = tk.Button(
            frame,
            text="📁 Upload Handwriting Samples (1-5 images)",
            command=self.upload_for_enrollment,
            bg=COLOR_PRIMARY,
            fg="white",
            font=("Arial", 11, "bold"),
            height=2
        )
        upload_btn.pack(fill=tk.X, pady=5)
        
        # Images preview
        self.enroll_images_label = tk.Label(
            frame,
            text="No images loaded",
            bg="white",
            fg="gray"
        )
        self.enroll_images_label.pack(pady=10)
        
        # Enroll button
        self.enroll_btn = tk.Button(
            frame,
            text="Enroll User",
            command=self.perform_enrollment,
            bg=COLOR_MATCH,
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            state=tk.DISABLED
        )
        self.enroll_btn.pack(fill=tk.X, pady=5)
    
    def create_management_section(self, parent):
        """Create management UI section."""
        frame = tk.LabelFrame(
            parent,
            text="⚙ Model Management",
            font=("Arial", 14, "bold"),
            bg="white",
            padx=15,
            pady=15
        )
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Statistics
        self.stats_text = tk.Text(frame, height=6, font=("Courier", 10), bg="#ecf0f1")
        self.stats_text.pack(fill=tk.X, pady=10)
        
        # Refresh stats button
        refresh_btn = tk.Button(
            frame,
            text="🔄 Refresh Statistics",
            command=self.update_statistics,
            bg=COLOR_PRIMARY,
            fg="white",
            font=("Arial", 10)
        )
        refresh_btn.pack(fill=tk.X, pady=5)
        
        # Retrain button
        retrain_btn = tk.Button(
            frame,
            text="🔧 Retrain SqueezeNet (Triplet Loss)",
            command=self.retrain_model,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2
        )
        retrain_btn.pack(fill=tk.X, pady=10)
        
        # Help text
        help_text = tk.Label(
            frame,
            text="Retraining requires:\n• Min 3 users\n• Min 2 samples/user",
            font=("Arial", 9),
            bg="white",
            fg="gray",
            justify=tk.LEFT
        )
        help_text.pack(pady=5)
        
        # Initial stats update
        self.update_statistics()
    
    # ========================================================================
    # IDENTIFICATION LOGIC
    # ========================================================================
    
    def upload_for_identification(self):
        """Handle image upload for identification."""
        filepath = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Load and validate
        image, msg = load_image(filepath)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image:\n{msg}")
            return
        
        self.current_image_path = filepath
        
        # Show preview
        display_img = image.copy()
        display_img.thumbnail(PREVIEW_IMAGE_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        
        self.id_image_label.configure(image=photo, text="")
        self.id_image_label.image = photo  # Keep reference
        
        # Enable identify button
        self.identify_btn.configure(state=tk.NORMAL)
    
    def perform_identification(self):
        """Perform biometric identification."""
        if not self.current_image_path:
            return
        
        # Show loading
        self.identify_btn.configure(state=tk.DISABLED, text="Identifying...")
        self.root.update()
        
        try:
            # Load image
            image, _ = load_image(self.current_image_path)
            
            # Preprocess and extract features based on model
            model_type = self.selected_model.get()
            
            if model_type == "svm":
                preprocessed = preprocess_for_svm(image)
                features = extract_svm_features(preprocessed)
            else:
                preprocessed = preprocess_for_squeezenet(image)
                features = extract_squeezenet_features(preprocessed)
            
            # Identify
            user_id, user_name, distance, decision, confidence = identify_user(features, model_type)
            
            # Display result
            self.display_identification_result(user_name, distance, decision, confidence)
        
        except Exception as e:
            messagebox.showerror("Error", f"Identification failed:\n{str(e)}")
        
        finally:
            self.identify_btn.configure(state=tk.NORMAL, text="Identify User")
    
    def display_identification_result(self, user_name, distance, decision, confidence):
        """Display identification results."""
        # Clear previous result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Result header
        if decision == "MATCH":
            color = COLOR_MATCH
            icon = "✅"
            status_text = "MATCH FOUND"
        elif decision == "UNCERTAIN":
            color = COLOR_UNCERTAIN
            icon = "!"
            status_text = "UNCERTAIN"
        else:
            color = COLOR_UNKNOWN
            icon = "❌"
            status_text = "UNKNOWN USER"
        
        header = tk.Label(
            self.result_frame,
            text=f"{icon} {status_text}",
            font=("Arial", 16, "bold"),
            bg=color,
            fg="white",
            padx=10,
            pady=10
        )
        header.pack(fill=tk.X, pady=5)
        
        # Details
        details_frame = tk.Frame(self.result_frame, bg="white")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        if user_name:
            tk.Label(
                details_frame,
                text=f"User: {user_name}",
                font=("Arial", 12, "bold"),
                bg="white"
            ).pack(anchor=tk.W, pady=2)
        
        tk.Label(
            details_frame,
            text=f"Distance: {distance:.4f}",
            font=("Arial", 10),
            bg="white"
        ).pack(anchor=tk.W, pady=2)
        
        tk.Label(
            details_frame,
            text=f"Confidence: {confidence:.1f}%",
            font=("Arial", 10),
            bg="white"
        ).pack(anchor=tk.W, pady=2)
        
        # Confidence bar
        bar_frame = tk.Frame(details_frame, bg="#dcdcdc", height=20)
        bar_frame.pack(fill=tk.X, pady=5)
        bar_frame.pack_propagate(False)
        
        bar_width = int(confidence * 2)  # Scale to 200px max
        bar = tk.Frame(bar_frame, bg=color, width=bar_width, height=20)
        bar.pack(side=tk.LEFT)
    
    # ========================================================================
    # ENROLLMENT LOGIC
    # ========================================================================
    
    def upload_for_enrollment(self):
        """Handle multiple image uploads for enrollment."""
        filepaths = filedialog.askopenfilenames(
            title="Select Handwriting Samples (1-5)",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepaths:
            return
        
        if len(filepaths) > MAX_SAMPLES_PER_USER:
            messagebox.showwarning(
                "Too Many Images",
                f"Maximum {MAX_SAMPLES_PER_USER} images allowed. Using first {MAX_SAMPLES_PER_USER}."
            )
            filepaths = filepaths[:MAX_SAMPLES_PER_USER]
        
        # Load all images
        self.current_images = []
        for filepath in filepaths:
            image, msg = load_image(filepath)
            if image is None:
                messagebox.showwarning("Warning", f"Skipped {os.path.basename(filepath)}:\n{msg}")
            else:
                self.current_images.append(image)
        
        if len(self.current_images) == 0:
            messagebox.showerror("Error", "No valid images loaded")
            return
        
        # Update label
        self.enroll_images_label.configure(
            text=f"{len(self.current_images)} image(s) loaded",
            fg="green"
        )
        
        # Enable enroll button if name is also provided
        if self.name_entry.get().strip():
            self.enroll_btn.configure(state=tk.NORMAL)
    
    def perform_enrollment(self):
        """Perform user enrollment."""
        # Validate name
        name = self.name_entry.get().strip()
        is_valid, result = validate_user_name(name)
        
        if not is_valid:
            messagebox.showerror("Invalid Name", result)
            return
        
        sanitized_name = result
        
        # Check if images loaded
        if len(self.current_images) == 0:
            messagebox.showerror("Error", "No images loaded")
            return
        
        # Show loading
        self.enroll_btn.configure(state=tk.DISABLED, text="Enrolling...")
        self.root.update()
        
        try:
            model_type = self.selected_model.get()
            
            # Extract features for all images
            svm_features_list = []
            squeezenet_features_list = []
            
            for image in self.current_images:
                # SVM features
                svm_preprocessed = preprocess_for_svm(image)
                svm_feat = extract_svm_features(svm_preprocessed)
                svm_features_list.append(svm_feat)
                
                # SqueezeNet features
                sqz_preprocessed = preprocess_for_squeezenet(image)
                sqz_feat = extract_squeezenet_features(sqz_preprocessed)
                squeezenet_features_list.append(sqz_feat)
            
            # Check similarity warning (use first sample)
            has_warning, similar_user, distance = check_similarity_warning(
                svm_features_list[0] if model_type == "svm" else squeezenet_features_list[0],
                model_type
            )
            
            if has_warning:
                response = messagebox.askyesno(
                    "! Similarity Warning",
                    f"The new user's handwriting is very similar to existing user:\n\n"
                    f"'{similar_user}' (distance: {distance:.4f})\n\n"
                    f"This could indicate:\n"
                    f"• Duplicate enrollment (same person)\n"
                    f"• Very similar writing styles\n\n"
                    f"Continue with enrollment anyway?"
                )
                
                if not response:
                    self.enroll_btn.configure(state=tk.NORMAL, text="Enroll User")
                    return
            
            # Enroll
            success, msg = enroll_user(
                sanitized_name,
                self.current_images,
                svm_features_list,
                squeezenet_features_list
            )
            
            if success:
                messagebox.showinfo(
                    "Success",
                    f"User '{sanitized_name}' enrolled successfully!\n"
                    f"User ID: {msg}\n"
                    f"Samples: {len(self.current_images)}"
                )
                
                # Reset
                self.name_entry.delete(0, tk.END)
                self.current_images = []
                self.enroll_images_label.configure(text="No images loaded", fg="gray")
                self.update_statistics()
            else:
                messagebox.showerror("Enrollment Failed", msg)
        
        except Exception as e:
            messagebox.showerror("Error", f"Enrollment failed:\n{str(e)}")
        
        finally:
            self.enroll_btn.configure(state=tk.NORMAL, text="Enroll User")
    
    # ========================================================================
    # MANAGEMENT LOGIC
    # ========================================================================
    
    def update_statistics(self):
        """Update database statistics display."""
        try:
            stats = get_database_stats()
            
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            
            text = f"""
Database Statistics
{'='*30}
Total Users:      {stats['total_users']}
Total Samples:    {stats['total_samples']}
Avg Samples/User: {stats['avg_samples_per_user']:.1f}
"""
            
            self.stats_text.insert(1.0, text)
            self.stats_text.config(state=tk.DISABLED)
        
        except Exception as e:
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error loading stats:\n{str(e)}")
            self.stats_text.config(state=tk.DISABLED)
    
    def retrain_model(self):
        """Retrain SqueezeNet model."""
        response = messagebox.askyesno(
            "Retrain SqueezeNet",
            "This will fine-tune SqueezeNet using triplet loss.\n\n"
            "Requirements:\n"
            f"• Minimum {MIN_USERS_FOR_RETRAIN} users\n"
            f"• Minimum {MIN_SAMPLES_FOR_RETRAIN} samples per user\n\n"
            "This may take several minutes. Continue?"
        )
        
        if not response:
            return
        
        # Run in background thread
        thread = threading.Thread(target=self._retrain_thread)
        thread.daemon = True
        thread.start()
    
    def _retrain_thread(self):
        """Background thread for retraining."""
        try:
            # Prepare data
            users = get_all_users()
            users_data = []
            
            for user in users:
                user_id = user["user_id"]
                
                # Load all images for this user
                images = []
                from layers.data_acquisition import get_user_sample_ids, load_encrypted_image
                
                sample_ids = get_user_sample_ids(user_id, DATABASE_ROOT)
                for sample_id in sample_ids:
                    img, _ = load_encrypted_image(user_id, sample_id, DATABASE_ROOT)
                    if img:
                        # Preprocess
                        preprocessed = preprocess_for_squeezenet(img)
                        images.append(preprocessed)
                
                if len(images) >= MIN_SAMPLES_FOR_RETRAIN:
                    users_data.append((user_id, images))
            
            # Retrain
            success, msg = retrain_squeezenet(users_data)
            
            # Show result
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model retrained successfully!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Retraining failed:\n{msg}"))
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Retraining error:\n{str(e)}"))
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def show_model_info(self):
        """Show information about models."""
        messagebox.showinfo(
            "Model Information",
            MODEL_COMPARISON_INFO
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

def launch_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = HBBIS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
