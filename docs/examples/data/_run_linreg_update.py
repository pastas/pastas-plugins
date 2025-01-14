def run():
    from numpy import linspace
    from pandas import Index, Series, read_csv

    parameters = read_csv("parameters_sel.csv", index_col=0)
    print("parameters view\n", parameters)
    # simulation = read_csv("simulation.csv")
    # x = simulation.index
    x = linspace(0.0, 1.0, 101)
    y = parameters.at["a", "optimal"] * x + parameters.at["b", "optimal"]
    simulation_new = Series(y, index=Index(x, name="x"), name="Simulation")
    simulation_new.to_csv("simulation.csv")
