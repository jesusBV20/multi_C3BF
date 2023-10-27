#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
- This script can configure and launch every simulation contained in the "simularions" directory -
"""

import os
import argparse

from simulations.utils.toolbox import createDir

# Import the simulation frames
from simulations.sim_0 import sim_0
from simulations.sim_1 import sim_1
from simulations.sim_2 import sim_2
from simulations.sim_3 import sim_3
from simulations.sim_4_intro import sim_4

# Create the output directory
FOLDER_OUTPUT = os.path.join("..", "output")
createDir(FOLDER_OUTPUT)

# DPI resolution dictionary
RES_DIC = {
    "480p"   : 640,
    "HD"     : 1280,
    "FullHD" : 1920,
    "2K"     : 2560,
    "4K"     : 3880
    }

"""\
Script call examples: 
    - python3 sim_launcher.py -id 0 -tf 70 -r 1 -g 1
    - python3 sim_launcher.py -id 1 -tf 70 -n 23 -r 1 -g 1
    - python3 sim_launcher.py -id 2 -tf 70 -n 50 -r 0.6 -g 1
    - python3 sim_launcher.py -id 3 -tf 70 -n 100 -no 50 -r 0.6 -g 1
    
"""
if __name__ == '__main__':
    # Parse script arguments
    parser = argparse.ArgumentParser(description="Circular formation")
    parser.add_argument('-id', '--id', dest='sim_id', type=int, default=1, help="ID of the simulation to be launches.")
    parser.add_argument('-tf', '--tf', dest='tf', type=float, default=None, help="Total time interval of the simulation integration.")
    parser.add_argument('-n', '--n', dest='n_agents', type=int, default=None, help="CBF parameter: n_agents")
    parser.add_argument('-no', '--no', dest='n_obs', type=int, default=None, help="CBF parameter: n_obs")
    parser.add_argument('-r', '--r', dest='r', type=float, default=None, help="CBF parameter: r")
    parser.add_argument('-g', '--g', dest='gamma', type=float, default=None, help="CBF parameter: gamma")
    parser.add_argument('-d', '--d', dest='d', type=float, default=None, help="CBF parameter: d")
    parser.add_argument('-tcbf', '--tcbf', dest='t_cbf', type=float, default=None, help="CBF parameter: t_cbf")
    parser.add_argument('-s', '--s', dest='s', type=float, default=None, help="GVF parameter: s")

    args = parser.parse_args()
    sim_id = args.sim_id

    # Revise the simulation arguments
    delattr(args, 'sim_id')
    sim_params = vars(args)

    field_to_pop = []
    for key, value in sim_params.items():
        if value is None:
            field_to_pop.append(key)

    for key in field_to_pop:
        sim_params.pop(key)

    print("Simulation parameters: ", sim_params)

    # Initilise the simulation frame
    if sim_id == 0:
        sim_frame = sim_0(**sim_params)
    elif sim_id == 1:
        sim_frame = sim_1(**sim_params)
    elif sim_id == 2:
        sim_frame = sim_2(**sim_params)
    elif sim_id == 3:
        sim_frame = sim_3(**sim_params)
    elif sim_id == 4:
        sim_frame = sim_4(**sim_params)
    else:
        print("ERROR: {0} is not a valid simulation frame ID!!".format(sim_id))
        sim_frame = None

    # Launch the numerical simulation and plot the summary
    if sim_frame is not None:
        print("Executing numerical simulation...")
        sim_frame.numerical_simulation()
        print("Plotting summary...")
        sim_frame.plot_summary(FOLDER_OUTPUT)

    # Generate the animation
    if sim_frame is not None:
        print("Generating the animation...")
        sim_frame.generate_animation(FOLDER_OUTPUT)
