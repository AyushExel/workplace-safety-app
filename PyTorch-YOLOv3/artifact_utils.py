import wandb


def init_new_run(name,job):
    run = wandb.init(project="artifact-workplace-safety",job_type=job,name=name)
    return run

def create_dataset_artifact(path,run,name):
    artifact = wandb.Artifact(name,type='dataset')
    artifact.add_dir(path)

    run.use_artifact(artifact)


def create_model_artifact(path,run,name):
    artifact = wandb.Artifact(name,type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)


#def create_result_artifact(path):
