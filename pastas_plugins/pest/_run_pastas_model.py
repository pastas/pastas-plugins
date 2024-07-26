def run():
    # load packages
    from pathlib import Path

    from pandas import read_csv
    from pastas.io.base import load as load_model

    # base path
    fpath = Path(__file__).parent

    # load pastas model
    ml = load_model(fpath / "model.pas")

    # update model parameters
    parameters = read_csv(fpath / "parameters_sel.csv", index_col=0)
    for pname, val in parameters.loc[:, "optimal"].items():
        ml.set_parameter(pname.replace("_g", "_A"), optimal=val)

    # simulate
    simulation = ml.simulate()
    simulation.loc[ml.observations().index].to_csv(fpath / "simulation.csv")
