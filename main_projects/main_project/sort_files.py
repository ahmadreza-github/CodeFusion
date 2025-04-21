# You pass a folder's path to "base_path" and run the script, different types(based on extension) of file will be put into folders with proper names.
import os
import shutil
from pathlib import Path
base_path = Path(r"C:\Users\AHMAD\OneDrive\Desktop\folder")
dotpy_path = base_path / '.PY file folder'
dotjson_path = base_path / '.JSON files folder'
csv_path = base_path / '.CSVs files folder'
rtf_path = base_path / '.RTF files folder'
pdf_path = base_path / '.PDF files folder'
txt_path = base_path / '.TXT files folder'
voice_music_path = base_path / 'Voice and music folder'
images_path = base_path / 'Image files folder'
video_path = base_path / "Video files folder"


def create_folder(end_with, path):
    global path_list
    path_list = os.listdir(base_path)

    # Check if any files end with the specified extension
    if any(file.endswith(end_with) for file in path_list):

        path.mkdir(parents=True, exist_ok=True)
        print(f'Folder created: {path}')
    else:
        print(f'No files ending with {end_with} found.')

def transfer_files(file_extension, file_type, target_path):
    path = base_path
    file_type = list(path.glob(file_extension))
    for files in file_type:
        shutil.move(files, target_path)
create_folder('.py', dotpy_path)
transfer_files(file_extension="*.py", file_type="python_list",
               target_path=dotpy_path)

create_folder('.txt', txt_path)
transfer_files(file_extension="*.txt",
               file_type="text_list", target_path=txt_path)

create_folder('.csv', csv_path)
transfer_files(file_extension="*.csv",
               file_type="csv_files", target_path=csv_path)

create_folder('.json', dotjson_path)
transfer_files(file_extension="*.json", file_type="json_files",
               target_path=dotjson_path)


create_folder('.pdf', pdf_path)
transfer_files(file_extension="*.pdf",
               file_type="pdf_list", target_path=pdf_path)


create_folder('.rtf', rtf_path)
transfer_files(file_extension="*.rtf",
               file_type="rtf_files", target_path=rtf_path)

create_folder(".mp3", voice_music_path)
transfer_files(file_extension="*.mp3",
               file_type="voice and music", target_path=voice_music_path)

create_folder(".png", images_path)
transfer_files(file_extension="*.png",
               file_type="images", target_path=images_path)

create_folder(".mp4", video_path)
transfer_files(file_extension="*.mp4",
               file_type="videos", target_path=video_path)
