import wandb


def init_new_run(name):
    wandb.init(project="workplace-safety-artifact",name=name)

def create_dataset_artifact(path,prefix):
    artifact = wandb.Artifact('demo-dataset', type='dataset')
    artifact.add_dir(path, name=prefix)

def create_model_artifact(model):
    artifact = wandb.Artifact('demo-model', type='model')
    artifact.add_file(path, name=prefix)

def create_result_artifact(path):




    
