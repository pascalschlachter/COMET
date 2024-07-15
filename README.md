# COMET: Contrastive Mean Teacher for Online Source-free Universal Domain Adaptation

This is the official repository to the paper ["COMET: Contrastive Mean Teacher for Online Source-free Universal Domain Adaptation"](https://arxiv.org/abs/2401.17728). Some parts of this implementation are based on [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation) and [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast).

## Usage
### Preparation
- Clone this repository
- Install the requirements by running `pip install -r requirements.txt`

### Source training
We uploaded the checkpoints of our pre-trained source models. To still do the source training yourself, edit the corresponding config file [source_training.yaml](configs/source_training.yaml) accordingly and run the following command: `python main.py fit --config configs/source_training.yaml`

### Source-only testing
To test without adaptation, i.e. to get the source-only results, edit the corresponding config file [source_only_testing.yaml](configs/source_only_testing.yaml) to select the desired scenario and run the following command: `python main.py test --config configs/source_only_testing.yaml`

### Online test-time adaptation to the target domain
We provide the config files for all domain and category shift scenarios in the folder [configs](configs). To perform the online test-time adaptation, select the config file corresponding the your desired scenario and run the following command: `python main.py fit --config configs/.../selected_config_file.yaml`
