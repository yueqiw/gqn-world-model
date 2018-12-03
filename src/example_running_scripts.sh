python run_gqn_face3d.py \
--data_dir /home/yueqi/Data1/data_ml/gqn_dataset/face3d_20181111-032039/train/ \
--gradient_steps 10000 \
--batch_size 20 \
--save_every 500 \
--print_every 500 \
--output_dir ../output


python run_gqn_unity.py \
--data_dir /home/yueqi/EVO/gqn_dataset/HallwayCamera2/train \
--model_name HallwayCamera \
--output_dir ../output \
--n_timesteps 10 \
--gradient_steps 20000 \
--batch_size 20 \
--save_every 500 \
--print_every 500


    
python run_gqn_unity.py \
--data_dir /home/yueqi/EVO/gqn_dataset/PyramidsCamera2/train \
--model_name PyramidsCamera \
--output_dir ../output \
--n_timesteps 10 \
--gradient_steps 20000 \
--batch_size 20 \
--save_every 500 \
--print_every 500


python run_gqn_unity.py \
--data_dir /home/yueqi/EVO/gqn_dataset/PushBlockCamera2/train \
--model_name PushBlockCamera \
--output_dir ../output \
--n_timesteps 10 \
--gradient_steps 20000 \
--batch_size 20 \
--save_every 500 \
--print_every 500

