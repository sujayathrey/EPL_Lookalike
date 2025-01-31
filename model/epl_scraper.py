import os
import time
import requests
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import model.epl_player_info as epl_player_info
import mediapipe as mp

# Initialize WebDriver
driver = webdriver.Chrome()

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Create folder for images
output_folder = "EPL_Player_Images"
os.makedirs(output_folder, exist_ok=True)

# Set up a set to keep track of downloaded image URLs or hashes
downloaded_hashes = set()

# Function to compute the hash of an image
def compute_image_hash(image_data):
    import hashlib
    return hashlib.sha256(image_data).hexdigest()

# Function to check if an image contains exactly one face using Mediapipe
def contains_exactly_one_face(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        # Check if exactly one face is detected
        return results.detections is not None and len(results.detections) == 1
    except Exception as e:
        print(f"Error during face detection: {e}")
        return False

# Function to download and validate an image
def download_image(img_url, player_folder, count):
    try:
        img_data = requests.get(img_url).content

        # Check if the image has already been downloaded
        img_hash = compute_image_hash(img_data)
        if img_hash in downloaded_hashes:
            print(f"Skipping duplicate image: {img_url}")
            return False
        downloaded_hashes.add(img_hash)

        # Load the image into OpenCV for validation
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Validate the presence of exactly one face
        if image is None:
            return False
        if not contains_exactly_one_face(image):
            return False

        # Save the valid image
        os.makedirs(player_folder, exist_ok=True)
        img_filename = os.path.join(player_folder, f"{count}.jpg")
        with open(img_filename, "wb") as img_file:
            img_file.write(img_data)

        print(f"Downloaded image: {img_filename}")
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

# Function to scrape images from Getty Images
def scrape_images(player_name):
    player_folder = os.path.join(output_folder, player_name.replace(" ", "_"))
    os.makedirs(player_folder, exist_ok=True)

    count = len(os.listdir(player_folder))  # Start with already downloaded images
    seen_urls = set()  # Track URLs to avoid re-downloading

    search_url = f"https://www.gettyimages.com/photos/{player_name.replace(' ', '+')}+portrait"

    driver.get(search_url)
    time.sleep(2)  # Allow time for the page to load

    while True:
        # Find all image elements currently visible
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img[src]")

        new_images_found = False  # Flag to track if new images are found

        # Iterate through image elements and download them
        for img in image_elements:
            img_url = img.get_attribute("src")
            if img_url and img_url not in seen_urls:  # Avoid duplicates
                seen_urls.add(img_url)  # Mark URL as seen
                if download_image(img_url, player_folder, count):
                    count += 1  # Increment count only for successfully downloaded images
                    new_images_found = True

        # Scroll slightly to load more images
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)  # Wait for the page to load new images

        # Stop if no new images were found
        if not new_images_found:
            print(f"Finished downloading images for {player_name}")
            break

# Loop through teams and players
for team, players in epl_player_info.epl_team_players.items():
    for player in players:
        print(f"Scraping images for {player}")
        scrape_images(player)

# Close WebDriver
driver.quit()
