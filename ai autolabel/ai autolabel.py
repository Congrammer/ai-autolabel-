import cv2
import os
import shutil
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import time
import random
import pytube
import zipfile

video1_name = input("title:")
channel_name = input("channel name:")
#num_videos =input("number of videos to download together:") 

def findvideo():
    num_videos = 1
    # Impostare il percorso della cartella di destinazione
    script_dir = os.path.dirname(__file__)
    download_path = os.path.join(script_dir, "download")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    # Impostare il percorso del file di registro
    log_file = os.path.join(script_dir, "log.txt")
    # Verificare se il file di registro esiste già e creare una lista di nomi di file scaricati
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            downloaded_files = f.read().splitlines()
    else:
        downloaded_files = []

    # Effettuare la ricerca del video su YouTube
    video_list = pytube.Search(video1_name + " " + channel_name).results
    video_list = [v for v in video_list if v.title not in downloaded_files]
    video_list = video_list[:num_videos]
    if not video_list:
        print("Nessun video trovato con il nome specificato")
        return

    # Scaricare i video trovati
    for video1 in tqdm(video_list, desc="Download progress", unit="video"):
        # Scaricare il video
        yt = pytube.YouTube(video1.watch_url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
        stream.download(download_path)
        # Aggiungere il nome del video scaricato al file di registro
        with open(log_file, "a") as f:
            f.write(video1.title + "\n")





def videoTOframe():
    # check if CUDA is available
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    # Impostare il percorso della cartella di destinazione
    script_dir = os.path.dirname(__file__)
    # set the folder containing the videos
    videos_folder = os.path.join(script_dir, "download")
    # set the desired frame rate for the extracted frames
    fps = 200
    # create a folder to save the extracted frames
    frames_folder_name=os.path.join(script_dir, "frames_extracted")
    if not os.path.exists(frames_folder_name):
        os.makedirs(frames_folder_name)

    frame_counter = 0

    # initialize counters for processed and total videos
    processed_videos = 0
    total_videos = len([filename for filename in os.listdir(videos_folder) if filename.endswith(".mp4") or filename.endswith(".avi")])
    # create a GPU context
    if use_cuda:
        ctx = cv2.cuda.Stream()
    # iterate over the files in the videos folder
    for filename in os.listdir(videos_folder):
        # check if the file is a video
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            # open the video using OpenCV
            video_path = os.path.join(videos_folder, filename)
            video = cv2.VideoCapture(video_path)
            # get the maximum width and height of the video
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # set the maximum resolution for frame capture
            video.set(cv2.CAP_PROP_FRAME_WIDTH, width/2)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, height/2)
            # get the total number of frames in the video
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # calculate the coordinates of the center of the video
            center_x = int(width / 2)
            center_y = int(height / 2)
            # calculate the dimensions of the extracted frame
            frame_size = min(width, height, 640)
            x = center_x - int(frame_size / 2)
            y = center_y - int(frame_size / 2)
            # extract frames from the video until the end
            frames_extracted = 0
            while True:
                # read the next frame from the video
                success, frame = video.read()
                # if there are no more frames to read, exit the loop
                if not success:
                    break
                # crop the frame to the desired size
                cropped_frame = frame[y:y+frame_size, x:x+frame_size]
                # save the frame as an image in the frames_extracted folder
                frame_name = f'{frames_folder_name}/{os.path.splitext(filename)[0]}_{frames_extracted}.jpg'
                cv2.imwrite(frame_name, cropped_frame)
                # increment the frame counter
                frames_extracted += 1
                # calculate the completion percentage
                progress = int(frames_extracted / frame_count * 100)
                remaining = 100 - progress
                # print the completion percentage
                print(f'Processing video {processed_videos+1}, Completion: {progress}%')
                # set a delay between each displayed frame to achieve the desired frame rate of 2 fps
                cv2.waitKey(int(1/fps))
        # release the resources of the current video
        video.release()

        # increment the processed video counter
        processed_videos += 1
    # Remove the input folder and all its contents
    shutil.rmtree(videos_folder)
    # close all windows
    cv2.destroyAllWindows()




def verificaimg():
    script_dir = os.path.dirname(__file__)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    confidence_threshold = 0.77
    input_folder = os.path.join(script_dir, "frames_extracted")
    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        results = model(image)
        detections = results.xyxy[0]
        keep_image = False
        for detection in detections:
            if detection[5] == 0:
                if detection[4] >= confidence_threshold:
                    keep_image = True
                    break

        if not keep_image:
            os.remove(image_path)
    


def crealabel():
    script_dir = os.path.dirname(__file__)
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.pt')
    confidence_threshold = 0.70
    model.maxdet = 10

    input_folder = os.path.join(script_dir, "frames_extracted")
    output_folder = os.path.join(script_dir, 'train')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    output_label_folder = os.path.join(output_folder, 'labels')
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # Wrap the loop with tqdm to show progress bar
    for filename in tqdm(os.listdir(input_folder), desc='Processing images'):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        results = model(image)
        detections = results.xyxy[0]
        # Create a new image object for drawing bounding boxes
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        # Create a text file for storing the labels, only if detections are found above confidence threshold
        if detections[detections[:, 4] >= confidence_threshold].any():
            label_path = os.path.join(output_label_folder, os.path.splitext(filename)[0] + '.txt')
            label_file = open(label_path, 'w')
            # Loop through all the detections in the image and write them to the same label file
            for detection in detections:
                if detection[5] == 0 and detection[4] >= confidence_threshold:  # check confidence score
                    # Draw a bounding box around the detected object
                    x1, y1, x2, y2 = detection[:4]
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    # Write the label to the label file
                    label = '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        (x1 + x2) / 2 / image.width,
                        (y1 + y2) / 2 / image.height,
                        (x2 - x1) / image.width,
                        (y2 - y1) / image.height
                    )
                    label_file.write(label)

                # Save the annotated image
                output_path = os.path.join(output_image_folder, filename)
                draw_image.save(output_path)

        if detections[detections[:, 4] >= confidence_threshold].any():
            label_file.close()
        else:
            # Remove the label file if no detections were found above confidence threshold
            label_path = os.path.join(output_label_folder, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(label_path):
                os.remove(label_path)

        # Remove the original image if no detections were found above confidence threshold
        if not detections[detections[:, 4] >= confidence_threshold].any():
            os.remove(image_path)
    # Remove the input folder and all its contents
    shutil.rmtree(input_folder)

def crea():
    script_dir = os.path.dirname(__file__)
    # Definisci i percorsi delle cartelle di train e valid
    train_path =  os.path.join(script_dir, 'train')
    valid_path =  os.path.join(script_dir, 'valid')
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    # Copia le cartelle "images" e "labels" nella cartella "valid"
    shutil.copytree(os.path.join(train_path, "images"), os.path.join(valid_path, "images"))
    shutil.copytree(os.path.join(train_path, "labels"), os.path.join(valid_path, "labels"))

    # Seleziona il 80% delle immagini da eliminare dalla cartella "valid/images"
    image_files = os.listdir(os.path.join(valid_path, "images"))
    num_images = len(image_files)
    num_valid_images = int(num_images * 0.8)
    valid_image_files = random.sample(image_files, num_valid_images)

    # Elimina le immagini selezionate dalla cartella "valid/images" e le corrispondenti label
    for filename in valid_image_files:
        image_path = os.path.join(valid_path, "images", filename)
        label_path = os.path.join(valid_path, "labels", filename[:-4] + ".txt")
        os.remove(image_path)
        os.remove(label_path)

    # Controlla le label rimanenti e rimuovi quelle senza immagini corrispondenti
    for label_filename in os.listdir(os.path.join(valid_path, "labels")):
        image_filename = label_filename[:-4] + ".jpg"
        image_path = os.path.join(valid_path, "images", image_filename)
        if not os.path.exists(image_path):
            label_path = os.path.join(valid_path, "labels", label_filename)
            os.remove(label_path)

    # Define the path of the "dataset" folder
    dataset_path = os.path.join(script_dir, 'dataset')
    # Create the "dataset" folder if it doesn't exist already
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Move the "train" and "valid" folders inside the "dataset" folder
    shutil.move(train_path, os.path.join(dataset_path, "train"))
    shutil.move(valid_path, os.path.join(dataset_path, "valid"))

    # Definisci la soglia minima di byte size per mantenere la cartella rinominata
    MIN_BYTE_SIZE = 1000
    # Leggi il numero progressivo dal file o inizializzalo a 1 se non esiste
    try:
        with open("counter.txt", "r") as f:
            counter = int(f.read())
    except FileNotFoundError:
        counter = 1
    # Rinomina la cartella "dataset" con il numero progressivo
    new_name = f"dataset_{counter:04d}"
    os.rename("dataset", new_name)
    # Elimina la cartella rinominata nell'ultima iterazione se il byte size è inferiore alla soglia
    if counter > 1:
        last_name = f"dataset_{(counter-1):04d}"
        last_path = os.path.join(os.getcwd(), last_name)
        if os.path.exists(last_path) and os.path.getsize(last_path) < MIN_BYTE_SIZE:
            os.rmdir(last_path)
    # Aggiorna il numero progressivo nel file
    with open("counter.txt", "w") as f:
        f.write(str(counter + 1))
    
while True:
    print('cercando video')
    findvideo()
    time.sleep(1)
    print('catturo frame')
    videoTOframe()
    time.sleep(1)
    print('verifico confidenza img')
    verificaimg()
    time.sleep(1)
    print('rilevo e creo label')
    crealabel()
    time.sleep(1)
    print('sto costruendo il dataset')
    crea()
    print('costruito')

