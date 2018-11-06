import torch
import os, gzip, io
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split pt files')
    parser.add_argument('--data_dir', type=str, help='data location', default="train")
    parser.add_argument('--out_dir', type=str, help='output location', default="train_scenes")

    args = parser.parse_args()

    filenames = [x for x in os.listdir(args.data_dir) if x.endswith(".pt.gz")]

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    for i in range(len(filenames)):
        print(i, filenames[i])
        gz_scene_path = os.path.join(args.data_dir, filenames[i])
        with gzip.open(gz_scene_path, 'rb') as f:
            # Use an intermediate buffer
            x = io.BytesIO(f.read())
            data = torch.load(x)
        
        gz_scene_folder = os.path.join(args.out_dir, filenames[i][:-6]) 
        if not os.path.exists(gz_scene_folder):
            os.mkdir(gz_scene_folder)
        
        for i, scenes in enumerate(data):
            scene_path = os.path.join(gz_scene_folder, "{}.pt.gz".format(i))
            with gzip.open(scene_path, 'wb') as f:
                torch.save(scenes, f)