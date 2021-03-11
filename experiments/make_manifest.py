import os
import json
import librosa

def build_manifest(audios_path, manifest_path ,total_files_count=None, skip_list_file=None):
    
    skip_list = []
    if skip_list_file:
        skip_list = open(skip_list_file).readlines()
        skip_list = [line.strip() for line in skip_list]
    
    curr_file_count = 0
    with open(manifest_path, 'w') as fout:
        for subdir, dirs, files in os.walk(audios_path):
            for file in files:
                if file in skip_list:
                    continue

                audio_file_path = os.path.join(subdir, file)
                transcript_file_path = audio_file_path.replace("wav","txt")
                
                if not os.path.isfile(transcript_file_path):
                    continue
                try:
                    transcript = open(transcript_file_path).readline().strip()
                    duration = librosa.core.get_duration(filename=audio_file_path)

                    curr_file_count += 1
                    if total_files_count and curr_file_count > total_files_count:
                        return
                    metadata = {
                        "audio_filepath": audio_file_path,
                        "duration": duration,
                        "text": transcript
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')
                    
                except:
                    pass
if __name__ == "__main__":
    #audios_path = "/home/prem/SSD/home/prem/nptel-data/wav"
    #manifest_path = "./train_manifest.json"
    #build_manifest(audios_path,manifest_path,total_files_count=50000,skip_list_file="./valid_files.list")

    audios_path = "/home/prem/SSD/home/prem/chosen-files/wav/"
    manifest_path = "./valid_with_corrected_labels_manifest.json"
    build_manifest(audios_path,manifest_path)
