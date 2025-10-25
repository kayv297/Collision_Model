'''
This scripts help automate the process of finding optimal gamma parameters for different config
'''
from find_gammas_graddesc import main as config_main
import os
import shutil

configs = [
    # {'n_reg': [3, 2], 'a_pos': [0, 1], 'name': 'D7_2'},
    # {'n_reg': [2, 2], 'a_pos': [0, 1, 2], 'name': 'D7_3'},
    # {'n_reg': [4, 4], 'a_pos': [0, 1], 'name': 'D10_2'},
    # {'n_reg': [4, 3], 'a_pos': [0, 1, 2], 'name': 'D10_3'},
    # {'n_reg': [3, 3], 'a_pos': [0, 1, 2, 3], 'name': 'D10_4'},
    # {'n_reg': [3, 2], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D10_5'},
    # {'n_reg': [4, 3], 'a_pos': [0, 1], 'name': 'D9_2'},
    # {'n_reg': [3, 3], 'a_pos': [0, 1, 2], 'name': 'D9_3'},
    # {'n_reg': [3, 2], 'a_pos': [0, 1, 2, 3], 'name': 'D9_4'},
    # {'n_reg': [3, 3], 'a_pos': [0, 1], 'name': 'D8_2'},
    # {'n_reg': [3, 2], 'a_pos': [0, 1, 2], 'name': 'D8_3'},
    # {'n_reg': [2, 2], 'a_pos': [0, 1, 2, 3], 'name': 'D8_4'},
    # {'n_reg': [2, 1], 'a_pos': [0, 1], 'name': 'D5_2'},
    # {'n_reg': [2, 2], 'a_pos': [0, 1], 'name': 'D6_2'},
    # {'n_reg': [2, 1], 'a_pos': [0, 1, 2], 'name': 'D6_3'},
    # {'n_reg': [5, 4], 'a_pos': [0, 1], 'name': 'D11_2'},
    # {'n_reg': [4, 4], 'a_pos': [0, 1, 2], 'name': 'D11_3'},
    # {'n_reg': [4, 3], 'a_pos': [0, 1, 2, 3], 'name': 'D11_4'},
    # {'n_reg': [3, 3], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D11_5'},
    # {'n_reg': [5, 5], 'a_pos': [0, 1], 'name': 'D12_2'},
    # {'n_reg': [5, 4], 'a_pos': [0, 1, 2], 'name': 'D12_3'},
    # {'n_reg': [4, 4], 'a_pos': [0, 1, 2, 3], 'name': 'D12_4'},
    {'n_reg': [4, 3], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D12_5'},
    {'n_reg': [3, 3], 'a_pos': [0, 1, 2, 3, 4, 5], 'name': 'D12_6'},
    {'n_reg': [6, 5], 'a_pos': [0, 1], 'name': 'D13_2'},
    {'n_reg': [5, 5], 'a_pos': [0, 1, 2], 'name': 'D13_3'},
    {'n_reg': [5, 4], 'a_pos': [0, 1, 2, 3], 'name': 'D13_4'},
    {'n_reg': [4, 4], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D13_5'},
    {'n_reg': [4, 3], 'a_pos': [0, 1, 2, 3, 4, 5], 'name': 'D13_6'},
    {'n_reg': [6, 6], 'a_pos': [0, 1], 'name': 'D14_2'},
    {'n_reg': [6, 5], 'a_pos': [0, 1, 2], 'name': 'D14_3'},
    {'n_reg': [5, 5], 'a_pos': [0, 1, 2, 3], 'name': 'D14_4'},
    {'n_reg': [5, 4], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D14_5'},
    {'n_reg': [4, 4], 'a_pos': [0, 1, 2, 3, 4, 5], 'name': 'D14_6'},
    {'n_reg': [4, 3], 'a_pos': [0, 1, 2, 3, 4, 5, 6], 'name': 'D14_7'},
    {'n_reg': [7, 6], 'a_pos': [0, 1], 'name': 'D15_2'},
    {'n_reg': [6, 6], 'a_pos': [0, 1, 2], 'name': 'D15_3'},
    {'n_reg': [6, 5], 'a_pos': [0, 1, 2, 3], 'name': 'D15_4'},
    {'n_reg': [5, 5], 'a_pos': [0, 1, 2, 3, 4], 'name': 'D15_5'},
    {'n_reg': [5, 4], 'a_pos': [0, 1, 2, 3, 4, 5], 'name': 'D15_6'},
    {'n_reg': [4, 4], 'a_pos': [0, 1, 2, 3, 4, 5, 6], 'name': 'D15_7'},
]

output_dir = '/home/tam/tam-workspace/playground/quantum-info/experiments/collision_model/data'
dest_dir = '/home/tam/tam-workspace/playground/quantum-info/experiments/collision_model/data/optimal'
for i, conf in enumerate(configs):
    print(f'[MAIN] Starting config ({i}/{len(configs)}): {conf["name"]}')
    try:
        config_main(conf['n_reg'], conf['a_pos'])
    except Exception:
        print(f"[Error] Error processing config: {conf}")
        continue

    # after processing the config, move data to optimal dir
    result_file = 'best_rounds.txt'
    checkpoint_file = 'checkpoint_progress.txt'

    result_path = f'{output_dir}/{result_file}'
    checkpoint_path = f'{output_dir}/{checkpoint_file}'

    # create a folder of current config: mkdir -p {dest_dir}/conf['name']
    # then move result_path and checkpoint_path to that folder

    config_folder = f"{dest_dir}/{conf['name']}"
    os.makedirs(config_folder, exist_ok=True)

    # Move files to the config folder
    if os.path.exists(result_path):
        shutil.move(result_path, f"{config_folder}/{result_file}")
    if os.path.exists(checkpoint_path):
        shutil.move(checkpoint_path, f"{config_folder}/{checkpoint_file}")

    print(
        f"[MAIN] Successfully processed and saved config ({i}/{len(configs)}): {conf['name']}")
