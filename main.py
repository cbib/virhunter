import fire
import classifier as clf


def train(
        ds, out, host, length,
        models=3,
        exp_name="None",
        n_cpus=1,
        n_gpus=0,
        epochs=10,
        batch_size=256,
):
    """
    ds - path to generated dataset
    out - path for output folder
    host - name of the plant, is used to load correct ds file
    length - length of encoding, is used to load correct ds file
    models - number of models to use
    exp_name - extra, used to separate experiments in wandb
    n_cpus - number of cpus available for training
    n_gpus - number of gpus available for training
    epochs - number of epochs
    batch_size - batch for training
    """
    clf.train.train(
        ds=ds,
        out=out,
        host=host,
        length=length,
        models=models,
        exp_name=exp_name,
        n_cpus=n_cpus,
        n_gpus=n_gpus,
        epochs=epochs,
        batch_size=batch_size,
    )


def predict(model, length, ds, out, n_cpus=4, batch_size=256):
    df_out = clf.predict.predict(
        model=model,
        length=length,
        ds=ds,
        n_cpus=n_cpus,
        batch_size=batch_size
    )
    df_out.to_csv(out)


if __name__ == '__main__':
    fire.Fire()
