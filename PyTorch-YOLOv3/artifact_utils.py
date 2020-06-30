import wandb


def init_new_run(name):
    run = wandb.init(project="artifact-workplace-safety",name=name)
    return run

def create_dataset_artifact(path,prefix,run):
    artifact = wandb.Artifact('demo-dataset', type='dataset')
    artifact.add_dir(path, name=prefix)
    run.log_artifact(artifact)


def create_model_artifact(path,run):
    artifact = wandb.Artifact('demo-model', type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)


#def create_result_artifact(path):
