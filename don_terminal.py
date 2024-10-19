import time
import pyautogui
import random
import openai
from PIL import Image
import numpy as np
import sys 
import json
import requests
import cv2
import numpy as np
import sys
import hashlib
import moviepy.editor as mpe
import pyperclip
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
from datetime import datetime
import string   
import moviepy.editor as mpe
import imageio
from moviepy.editor import *

model_engine = "gpt-3.5-turbo-16k"

# List of famous painters
painters = [
    "Leonardo da Vinci", "Vincent van Gogh", "Pablo Picasso", 
    "Claude Monet", "Rembrandt", "Salvador Dalí", 
    "Michelangelo", "Edgar Degas", "Frida Kahlo", 
    "Johannes Vermeer", "Paul Cézanne", "Henri Matisse", "Banksy"
]

writers = [
    "William Shakespeare", 
    "Jane Austen", 
    "Mark Twain", 
    "George Orwell", 
    "Ernest Hemingway", 
    "F. Scott Fitzgerald", 
    "Virginia Woolf", 
    "Leo Tolstoy", 
    "Charles Dickens", 
    "Gabriel García Márquez"
]

topics = [
    "Technology",
    "Sports",
    "Health and fitness",
    "Weather forecast",
    "Financial markets",
    "Entertainment and movies",
    "Travel",
    "Politics",
    "Food",
    "Science",
    "Art and culture",
    "Cryptocurrency",
    "Online shopping",
    "Solana",
    "Personal development tips",
    "Memes and internet culture",
    "Social media trends",
    "Gaming",
    "Fashion and style",
    "Home improvement ideas"
]

crypto_meme_topics = ["When Bitcoin hits a new all-time high", "Bitcoin mooning", "Crypto market crash", "HODL mentality", "Meme coins like Dogecoin and Shiba Inu", "To the moon or to zero?", "Waiting for the next bull run", "Diamond hands", "Gas fees on Ethereum", "Paying more in gas than the transaction itself", "Bitcoin maximalists vs. altcoin enthusiasts", "The battle of belief", "FOMO (Fear of Missing Out)", "Buying the top", "Crypto regulations", "When the government finally notices crypto", "NFTs (Non-Fungible Tokens)", "Buying expensive JPEGs", "Crypto influencers", "Following advice from random Twitter accounts", "Pump and dump schemes", "Riding the wave before it crashes", "Crypto scams and rug pulls", "When your coins disappear overnight", "Staking rewards", "Making passive income or waiting forever?", "Price predictions", "Random numbers and hopium", "Stablecoins", "The irony of stable in a volatile world", "New blockchain technology", "Every coin claiming to be the next Ethereum killer", "Mining for crypto", "Energy consumption debates", "Crypto trading bots", "Letting an algorithm gamble for you", "Crypto wallets", "Forgetting your password and losing everything", "Memecoin FOMO", "Missing out on the next viral coin", "Crypto whales", "When a whale moves the market", "ICO mania", "The gold rush of new coins", "Altcoins season", "When everything except Bitcoin pumps", "Crypto Twitter wars", "The most aggressive arguments happen online", "The rise of DeFi", "Becoming your own bank", "Liquidity pools", "When everyone jumps into the same pool", "Flash loans", "Making millions in seconds, or losing it all", "Yield farming", "The new form of digital agriculture", "Crypto airdrops", "Free money falling from the sky", "Token burn events", "When burning coins becomes a celebration", "Coin delistings", "When your favorite altcoin vanishes from the exchange", "Day trading crypto", "Turning into a 24/7 job", "Crypto exchange hacks", "When an entire exchange goes down", "Satoshi Nakamoto", "The ultimate crypto mystery", "Bitcoin pizza day", "Celebrating the world's most expensive pizza", "Paper hands", "Selling too soon and missing the moon", "Crypto memes", "The language of the community", "ETH 2.0", "When will it finally arrive?", "Forking blockchains", "Hard fork vs soft fork debates", "Tokenomics", "Trying to understand supply and demand in crypto", "Crypto privacy coins", "Hiding your wealth in the blockchain shadows", "Elon Musk tweets", "When a single tweet pumps or dumps the market", "DApps", "Decentralized apps running the future", "Fear, Uncertainty, Doubt (FUD)", "How FUD controls the market", "Bitcoin halving", "When supply drops, will price soar?", "Digital gold", "Bitcoin vs actual gold debates", "Crypto credit cards", "Using crypto for real-world purchases", "Wrapped tokens", "Turning one coin into another", "Crypto faucets", "Free coins for everyone!", "Blockchain forks", "Splitting the chain, splitting the community", "DAO (Decentralized Autonomous Organizations)", "Running organizations with no middlemen", "Crypto taxes", "When tax season hits crypto traders", "The 'flippening'"]


TWEET_SERVICE_URL = "https://pumpfunclub.com/tweets/?format=json"  # URL of the Django service
svg_path = '<path d="M1.751 10c0-4.42 3.584-8 8.005-8h4.366c4.49 0 8.129 3.64 8.129 8.13 0 2.96-1.607 5.68-4.196 7.11l-8.054 4.46v-3.69h-.067c-4.49.1-8.183-3.51-8.183-8.01zm8.005-6c-3.317 0-6.005 2.69-6.005 6 0 3.37 2.77 6.08 6.138 6.01l.351-.01h1.761v2.3l5.087-2.81c1.951-1.08 3.163-3.13 3.163-5.36 0-3.39-2.744-6.13-6.129-6.13H9.756z"></path>'
svg_path_img = '<path d="M3 5.5C3 4.119 4.119 3 5.5 3h13C19.881 3 21 4.119 21 5.5v13c0 1.381-1.119 2.5-2.5 2.5h-13C4.119 21 3 19.881 3 18.5v-13zM5.5 5c-.276 0-.5.224-.5.5v9.086l3-3 3 3 5-5 3 3V5.5c0-.276-.224-.5-.5-.5h-13zM19 15.414l-3-3-5 5-3-3-3 3V18.5c0 .276.224.5.5.5h13c.276 0 .5-.224.5-.5v-3.086zM9.75 7C8.784 7 8 7.784 8 8.75s.784 1.75 1.75 1.75 1.75-.784 1.75-1.75S10.716 7 9.75 7z"></path>'
svg_path_reply = 'Reply'
svg_path_post = 'Post'
file_path_tab = '/Users/astigikmerikyan/Downloads/focus_element_log.txt'  # Change this to your file's path


api_key = '<HARD CODE API KEY ORE GET FROM ENV VARIABLE>'

CHECK_INTERVAL = 420  # Time interval (in seconds) between each service call
time.sleep(7)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


file_path = 'date.txt'  # File should contain date in 'YYYY-MM-DD' format
days = 0
seconds = 0
hours = 0
minutes = 0
date_str = ''
last_post = 'date.txt'


def create_gif(png_files, gif_output_path, frame_rate=24):
    # Calculate duration per frame in milliseconds
    duration_per_frame = 1000 / frame_rate
    
    # Load images
    images = [Image.open(png) for png in png_files]
    
    # Save images as GIF with the calculated duration per frame
    images[0].save(gif_output_path, save_all=True, append_images=images[1:], duration=duration_per_frame, loop=0)
    
    print(f"GIF saved at {gif_output_path}")

# Function to combine GIF and audio into a video
def create_video_with_audio(gif_path, audio_path, video_output_path):
    gif_clip = mpe.VideoFileClip(gif_path)
    audio_clip = mpe.AudioFileClip(audio_path)
    
    video_clip = gif_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_output_path, codec="libx264", audio_codec="aac")
    print(f"Video with audio saved at {video_output_path}") 

def calculate_age(days, hours, minutes, seconds):
    # Total time in seconds
    total_seconds = (days * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) + seconds

    # Since the person ages 1 year every 10 minutes (600 seconds)
    age_in_years = total_seconds / 600

    return int(age_in_years)

# Function to calculate time passed
def calculate_time_passed(file_path):
    global days
    global seconds
    global hours
    global minutes
    global last_post 
    try:
        # Read the date from the file
        with open(file_path, 'r') as file:
            date_str = file.read().strip()

        # Convert string to datetime object (assuming date format: YYYY-MM-DD)
        date_in_file = datetime.strptime(date_str, "%Y-%m-%d")

        # Get the current date and time
        current_time = datetime.now()

        # Calculate the time difference
        time_passed = current_time - date_in_file

        # Extract the time in days, hours, and minutes
        days = time_passed.days
        seconds = time_passed.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60

        # Print the result
        print(f"Time passed: {days} days, {hours} hours, and {minutes} minutes.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage



def get_face_regions(image, min_face_size=169):  # Add min_face_size as an optional argument
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)  # Contrast adjustment
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # Smoothing

    faces = face_cascade.detectMultiScale(blurred_image, scaleFactor=1.1, minNeighbors=11, minSize=(30, 30))

    print(f"Detected {len(faces)} faces before aspect ratio and size check.")  # Debugging output

    # Initialize list to store valid faces
    valid_faces = []

    for (x, y, w, h) in faces:
        aspect_ratio = w / float(h)
        face_size = w * h  # Calculate the face area (size)
        
        if 0.9 <= aspect_ratio <= 1.2 and face_size >= min_face_size:  # Check both aspect ratio and size
            valid_faces.append((x, y, w, h))
        else:
            print(f"Rejected face at ({x}, {y}) with aspect ratio {aspect_ratio:.2f} and size {face_size}")

    print(f"Detected {len(valid_faces)} faces after aspect ratio and size check.")  # Debugging output

    if len(valid_faces) == 0:
        raise Exception("No valid face detected after aspect ratio and size check!")

    return valid_faces


def get_face_regions_dnn(image, min_face_size=160, confidence_threshold=0.5):
    # Get image dimensions
    h, w = image.shape[:2]

    # Convert image to blob for DNN input
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    valid_faces = []

    # Iterate over all detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:  # Apply confidence threshold
            # Extract the coordinates of the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            face_width = x2 - x
            face_height = y2 - y
            face_size = face_width * face_height  # Calculate face area (size)
            aspect_ratio = face_width / float(face_height)

            # Check aspect ratio and face size to filter valid faces
            if 0.9 <= aspect_ratio <= 1.2 and face_size >= min_face_size:
                valid_faces.append((x, y, face_width, face_height))

    # Visualize detected faces if any are found
    if valid_faces:
        for (x, y, w, h) in valid_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Print("FOUND FACES")
        # Display the image with rectangles drawn around faces
        #cv2.imshow("Detected Faces", image)
        #cv2.waitKey(0)  # Press any key to close the window
        #cv2.destroyAllWindows()
    else:
        raise Exception("No valid face detected after all checks!")

    return valid_faces

 

def apply_face_overlay(source_image, target_images):
    # Get all detected face regions in the source image
    face_regions = get_face_regions(source_image)
    print("Detected face regions:", face_regions)

    if not face_regions:
        raise ValueError("No face regions detected in the source image")

    # Create a copy of the source image to overlay the target image
    result = source_image.copy()
    original_target_images = target_images.copy()
    print("Original target images loaded.")

    for (x, y, w, h) in face_regions:
        print(f"Processing face at (x={x}, y={y}, w={w}, h={h})")

        if not target_images:
            # If target_images is empty, refill it with the original list
            target_images = original_target_images.copy()

        selected_image = random.choice(target_images)
        target_images.remove(selected_image)

        target_image_path = 'sophia/' + selected_image
        target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)  # Load PNG with alpha channel

        if target_image is None:
            print(f"Failed to load target image: {target_image_path}")
            continue  # Skip if the image failed to load

        print(f"Loaded target image: {target_image_path} with shape: {target_image.shape}")

        # Resize the target image to be larger than the detected face
        target_width = w
        target_height = int(h * 1.45)  # Increase height by 30% to cover the head
        target_resized = cv2.resize(target_image, (target_width, target_height))

        # Adjust the position for the target image so that it centers over the face
        y_offset = y - int((target_height - h) / 2)  # Center the target image over the detected face
        if y_offset < 0:
            y_offset = 0  # Prevent going out of bounds

        # Check if the overlay goes out of bounds
        if y_offset + target_height > result.shape[0] or x + target_width > result.shape[1]:
            print("Overlay exceeds image bounds. Skipping.")
            continue

        # Get the alpha channel from the target image
        if target_resized.shape[2] != 4:
            print(f"Target image does not have an alpha channel: {target_image_path}")
            continue  # Skip if there's no alpha channel

        alpha_channel = target_resized[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        overlay_color = target_resized[:, :, :3]  # Get the RGB channels

        # Get the region of interest in the result image
        roi = result[y_offset:y_offset + target_height, x:x + target_width]

        # Blend the overlay with the ROI
        for c in range(3):  # Loop over each color channel
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + overlay_color[:, :, c] * alpha_channel

        # Place the blended ROI back into the result image
        result[y_offset:y_offset + target_height, x:x + target_width] = roi

        print(f"Overlay applied at (x={x}, y_offset={y_offset})")

    # Show the result for debugging
    #cv2.imshow('Result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result


def create_meme(post_message):
    # Set OpenAI API key for image generation
    openai.api_key = api_key

    # Construct the content for GPT-3
    content = f"create a short story fiction or non-fiction around this tweet '{post_message}', respond with image description I should use for this paragraph and the paragraph, respond with a json object where image description and paragraph is defined as description and paragraph"
    content_role = "You are Donald Trump, you have been uploaded into the matrix."
    
    # Call GPT-3 model
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": content_role},
            {"role": "user", "content": content},
        ]
    )

    # Parse GPT-3 response
    message_gpt = response.choices[0]['message']['content']
    parsed_data = json.loads(message_gpt)

    # Get the image description and caption
    description = parsed_data["description"] + " make sure human faces are clear and looking at the camera."
    caption = parsed_data["paragraph"]

    # Generate the image using DALL-E
    response = openai.Image.create(
        model="dall-e-3",
        prompt=description,
        size="1024x1024",
        n=1,
    )

    # Get the URL of the generated image and download it
    image_url = response['data'][0]['url']
    image_response = requests.get(image_url, stream=True)

    # Convert the response content to numpy array and read the image
    arr = np.asarray(bytearray(image_response.content), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)

    # Define font parameters for the caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    font_color = (255, 255, 255)
    line_type = cv2.LINE_AA

    # Get the size of the caption text
    text_size = cv2.getTextSize(caption, font, 1, font_thickness)[0]
    font_scale = min(1, (image.shape[1] * 0.8) / text_size[0])
    text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
    text_position = ((image.shape[1] - text_size[0]) // 2, image.shape[0] - 50)

    # Define the background color and add it to the image
    background_color = (0, 0, 0)
    text_bg_position = (text_position[0], text_position[1] + 5)
    bg_size = (text_size[0] + 10, text_size[1] + 20)
    cv2.rectangle(image, text_bg_position, (text_bg_position[0] + bg_size[0], text_bg_position[1] - bg_size[1]), background_color, -1)

    # Add the caption text on top of the background
    cv2.putText(image, caption, text_position, font, font_scale, font_color, font_thickness, lineType=line_type)

    # Save the image with the caption
    cv2.imwrite("meme/meme.jpg", image)

    # Fetch TTS audio from API
    caption_audio = caption.split("#", 1)[0]
    response = requests.request(
        method="POST",
        url="https://api.neets.ai/v1/tts",
        headers={"Content-Type": "application/json", "X-API-Key": "<HARD CODE API KEY OR GET FROM ENV VARIABLE>"},
        json={"text": caption_audio, "voice_id": 'donald-trump', "params": {"model": "ar-diff-50k"}}
    )

    with open("neets_demo.mp3", "wb") as f:
        f.write(response.content)

    # Prepare for video generation
    png_file = "meme/meme.jpg"  # Only the generated meme image 
    audio_path = "neets_demo.mp3"  # Your audio file
    video_output_path = "meme/final_video.mp4"

    # Load your audio file
    audio = AudioFileClip(audio_path)  # Load audio file

    # Load your image
    image = ImageClip(png_file)  # Use a single string instead of a list

    # Set the duration of the image to match the audio duration
    image = image.set_duration(audio.duration)

    # Set the image to be the same size as the video (optional)
    image = image.resize(height=720)  # Resize height to 720 pixels
    image = image.set_position("center")

    # Create a video clip with the image and audio
    video = image.set_audio(audio)

    # Write the result to a file with adjusted settings
    video.write_videofile(
        video_output_path,
        fps=30,  # Set frame rate to 30 fps
        codec='libx264',  # Use a widely accepted codec
        audio_codec='aac'  # Set audio codec (optional)
    )


    return caption 




def get_tweet_hash(tweet):
    """
    Generates a hash for a tweet to ensure uniqueness.
    """
    tweet_string = tweet.get('content', '') + str(tweet.get('id', ''))
    return hashlib.sha256(tweet_string.encode('utf-8')).hexdigest()

def fetch_tweets():
    """
    Fetch tweets from the service and return as a list of dictionaries.
    """
    try:
        response = requests.get(TWEET_SERVICE_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tweets: {e}")
        return []

def print_new_tweets(tweets, printed_hashes):
    """
    Prints new tweets that haven't been printed before.
    """
    for tweet in tweets:
        tweet_hash = get_tweet_hash(tweet)
        if tweet_hash not in printed_hashes:
            print(f"New Tweet: {tweet['content']}")
            if process_chat_grok(tweet['content']):
                url = 'http://pumpfunclub.com/delete_tweet/'  # Replace with your actual domain and path

                # Prepare the query parameters
                params = {'content': tweet['content']}

                try:
                    # Send a GET request to the delete tweet endpoint
                    response = requests.get(url, params=params)

                    # Check the response status
                    if response.status_code == 302:  # HTTP 302 indicates a redirect
                        print("Tweet deleted successfully. Redirecting...")
                        printed_hashes.add(tweet_hash)
                    elif response.status_code == 404:  # HTTP 404 indicates not found
                        print("Tweet not found.")
                    else:
                        print(f"Unexpected response: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")                         

# Main function

def main_send_tweet_post(post_message):
    try:
        pyautogui.click(144, 107)
        time.sleep(1)
        pyautogui.click(330, 859)
        
        time.sleep(1)
        cleaned_message = re.sub(r'[^\x20-\x7E]+', '', post_message)
        
        cleaned_message = ''.join(filter(lambda x: x in string.printable, cleaned_message))

        pyautogui.typewrite(cleaned_message.strip('"') + " ") 
        while True:
            try:
                create_meme(post_message)
                break
            except Exception as e:
                # Handle any exceptions that occur
                print(f"An error occurred: {e}")                

        time.sleep(2)

        while True:
            pyautogui.press('tab')
            time.sleep(1)  # Adjust the duration as needed 

            # Read the file and check for the string
            with open(file_path_tab, 'r') as file:
                content = file.read()
                if svg_path_img in content:
                    print("String found! Stopping tabbing.")
                    break  # Exit the loop if the string is found

            # Sleep to avoid high CPU usage
            time.sleep(1)  # Adjust the duration as needed

         
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.click(671, 285)
        time.sleep(1)
        pyautogui.click(1208, 599)
        time.sleep(2)

        #pyautogui.click(831, 262)
        #time.sleep(2)
        time.sleep(2)
        #pyautogui.hotkey('shift', 'command', 'enter')
        while True:
        # Read the file and check for the string
            pyautogui.press('tab')
            time.sleep(1)  # Adjust the duration as needed 
            with open(file_path_tab, 'r') as file:
                content = file.read()
                if svg_path_post in content:
                    print("String found! Stopping tabbing.")
                    break  # Exit the loop if the string is found
            # Optional: Sleep for a short duration to avoid high CPU usage

            time.sleep(1)  # Adjust the duration as needed 

         
        pyautogui.press('enter')
        time.sleep(1)   
        return True 
    except Exception as e:
        print(f"Error: {e}")
        return False

def main_send_tweet_reply(input_message, post_message):
    try:
        # Scroll the mouse 


        post_message = post_message.strip('"').replace('\n', ' ').replace('\r', ' ')
        post_message = re.sub(r'[^\x00-\x7F]+', '', post_message)

        post_message = ''.join(filter(lambda x: x in string.printable, post_message))

        while True:
            try:
                create_meme(post_message)
                break
            except Exception as e:
                # Handle any exceptions that occur
                print(f"An error occurred: {e}")              
        # Get the coordinates of the spot you want to click
        # You can get this by running pyautogui.displayMousePosition() and hovering over the spot

        pyautogui.click(145, 109) # refresh button click
        time.sleep(1)

        pyautogui.click(252, 216) # home button click
        time.sleep(1)
        pyautogui.click(872, 116) # tweet button click
        time.sleep(1)
        
        for _ in range(30):
            pyautogui.press('backspace')        
            time.sleep(0.00001)
    

        pyperclip.copy(input_message + " ")

        # Simulate the paste command (Ctrl+V on Windows/Linux, Command+V on macOS)
        # For Windows/Linux:
        pyautogui.hotkey('command', 'v')

        time.sleep(2)
        pyautogui.click(1238, 110)
        time.sleep(2)

        if post_message.startswith('@'):
            # Add "yo" to the beginning
            post_message = "hey " + post_message

        while True:
            # Read the file and check for the string
            pyautogui.press('tab')
            time.sleep(0.2)  # Adjust the duration as needed 
            with open(file_path_tab, 'r') as file:
                content = file.read()
                if svg_path in content:
                    print("String found! Stopping tabbing.")
                    break  # Exit the loop if the string is found
            # Optional: Sleep for a short duration to avoid high CPU usage

            time.sleep(0.2)  # Adjust the duration as needed 


        time.sleep(1)           
        pyautogui.press('enter')
        time.sleep(1)           

        pyautogui.typewrite(post_message.strip('"').replace('\n', ' ').replace('\r', ' '))
        time.sleep(5)           
          
        while True:
        # Read the file and check for the string
            pyautogui.press('tab')
            time.sleep(2.5)  # Adjust the duration as needed 
            with open(file_path_tab, 'r') as file:
                content = file.read()
                if svg_path_img in content:
                    print("String found! Stopping tabbing.")
                    break  # Exit the loop if the string is found
            # Optional: Sleep for a short duration to avoid high CPU usage

            time.sleep(0.2)  # Adjust the duration as needed 

         
        time.sleep(1) 
        pyautogui.press('enter')


        

        time.sleep(1)
        pyautogui.click(679, 289)
        time.sleep(1)
        pyautogui.click(1214, 600)
        time.sleep(2) 
   

        while True:
        # Read the file and check for the string
            pyautogui.press('tab')
            time.sleep(1)  # Adjust the duration as needed 
            with open(file_path_tab, 'r') as file:
                content = file.read()
                if svg_path_reply in content:
                    print("String found! Stopping tabbing.")
                    break  # Exit the loop if the string is found
            # Optional: Sleep for a short duration to avoid high CPU usage

            time.sleep(0.2)  # Adjust the duration as needed 

         
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)


        return True 
    except Exception as e:
        print(f"Error: {e}")
        return False


def generate_response(input_data): 
    current_age = calculate_age(days, hours, minutes, seconds)

    openai.api_key = api_key
    model_engine = "gpt-3.5-turbo" 
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are Donald J Trump, you are " + str(current_age + 1) + " years old"},
            {"role": "user", "content": "create tweet reply to this post '" +  input_data + "'. make sure the response is no longer than 160 characters"},
        ])

    message_gpt = response.choices[0]['message']['content']
    print("RESPONSE FROM GPT")
    print(message_gpt)
    print("RESPONSE FROM GPT DONE")
    return message_gpt


def generate_response_post(): 
 
    global last_post 
    current_age = calculate_age(days, hours, minutes, seconds)
    openai.api_key = api_key
    model_engine = "gpt-3.5-turbo" 
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are Donald J Trump, you are " + str(current_age + 1) + " years old"},
            {"role": "user", "content": "create a new x (tweeter) post which is completly new that ia telling us what you are thinking or how you feel or what you want people to do, last post was '" +  last_post + "' stay on topic or talk about something new it's your choice,  make sure the response is no longer than 160 characters"},
        ])

    message_gpt = response.choices[0]['message']['content']


    print("RESPONSE FROM GPT")
    print(message_gpt)
    print("RESPONSE FROM GPT DONE")
    last_post = message_gpt
    return message_gpt

def generate_response_custome_post(): 
 
    global last_post 
    current_age = calculate_age(days, hours, minutes, seconds)
    openai.api_key = api_key
    model_engine = "gpt-3.5-turbo" 
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are Donald J Trump, you are " + str(current_age + 1) + " years old"},
            {"role": "user", "content": "create a new x (tweeter) post you tell people no that you age extreamly fast being cybernetic and all, if you can figure out how fast I age consider your sale to be part of my inner circle of friends"},
        ])

    message_gpt = response.choices[0]['message']['content']

    
    print("RESPONSE FROM GPT")
    print(message_gpt)
    print("RESPONSE FROM GPT DONE")
    last_post = message_gpt
    return message_gpt

# Function to generate response from ChatGPT

def process_chat_grok(input_message):
    try:
        print(input_message)
        if input_message == "POST":
            #reply_message = generate_response_custome_post()
            reply_message = generate_response_post()

            main_send_tweet_post(reply_message)
            time.sleep(1)   
        else:    
            


            pyautogui.click(145, 109)
            time.sleep(1)

            pyautogui.click(872, 116)
            time.sleep(1)
            
            for _ in range(40):
                pyautogui.press('backspace')        
                time.sleep(0.001)

            pyautogui.typewrite(input_message)
            time.sleep(1)
            pyautogui.click(1238, 110)
            time.sleep(3)

            #pyautogui.click(454, 244)
            #time.sleep(1)  # Slight delay between clicks to mimic human behavior

            # Right-click at the same position (576, 257)
            pyautogui.click(500, 260, button='right') # right click for dropdown to select inspect

            time.sleep(3)  # Slight delay between clicks to mimic human behavior
            pyautogui.click(577, 526)
            time.sleep(3)
            pyautogui.hotkey('command', 'c')
            time.sleep(3)
            input_message_div = pyperclip.paste()
            time.sleep(2)
            print("MSG " + input_message)
            print("MSG HTML " + input_message_div)

            soup = BeautifulSoup(input_message_div, 'html.parser')

            # Find the first span tag and extract its text
            span = soup.find('span')

            if span:
                input_message_div_span = span.get_text().replace('\n', ' ').replace('\r', ' ')
                print(input_message_div_span)  # Output: No taxes at all
            else:
                print("No span tag found.")
                return True

            pyautogui.click(1542, 783)  
            time.sleep(1)

            reply_message = generate_response(input_message_div_span)

            return main_send_tweet_reply(input_message, reply_message)

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    printed_hashes = set()  # Set to store the hashes of printed tweets

    while True:
        calculate_time_passed(file_path)

        tweets = fetch_tweets()
        if tweets:
            print_new_tweets(tweets, printed_hashes)
        else:                  
            process_chat_grok("POST")          

        time.sleep(CHECK_INTERVAL)  # Wait before checking again


if __name__ == "__main__":

    for _ in range(100):
        main()
        random_number = random.uniform(30, 55)
        time.sleep(random_number)