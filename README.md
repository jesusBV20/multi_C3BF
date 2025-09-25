# Behavioral-based circular formation control for robot swarms

This repository contains Python simulations designed to numerically validate the behavior-based circular formation control algorithm for robot swarms.

## Research Conference Paper

**ABSTRACT:** This paper focuses on coordinating a robot swarm orbiting a convex path without collisions among the individuals. The individual robots lack braking capabilities and can only adjust their courses while maintaining their constant but different speeds. Instead of controlling the spatial relations between the robots, our formation control algorithm aims to deploy a dense robot swarm that mimics the behavior of \emph{tornado} schooling fish. To achieve this objective safely, we employ a combination of a scalable overtaking rule, a guiding vector field, and a control barrier function with an adaptive radius to facilitate smooth overtakes. The decision-making process of the robots is distributed, relying only on local information. Practical applications include defensive structures or escorting missions with the added resiliency of a swarm without a centralized command. We provide a rigorous analysis of the proposed strategy and validate its effectiveness through numerical simulations involving a high density of unicycles.

    @article{jesusbv2024bcf,
      title={Behavioral-based circular formation control for robot swarms},
      author={Bautista, Jesús and de Marina, Héctor García},
      year={2024},
      booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
      pages={8989-8995},
      doi={10.1109/ICRA57147.2024.10610826}
    }

In the proceedings of the IEEE International Conference on Robotics and Automation (ICRA) 2024.

PDF: https://arxiv.org/abs/2309.09101

For more research on robot swarms, visit our research web: https://www.swarmsystemslab.eu/

## Installation

We recommend creating a dedicated virtual environment to ensure that the project dependencies do not conflict with other Python packages:
```bash
python -m venv venv
source venv/bin/activate
```
Then, install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ```requirements.txt``` contains the versions tested for **compatibility with the simulator**.
Do **not modify the versions** to ensure stable and reproducible environments. Note that ```ssl_simulator``` already provides stable versions for the following core packages: ```numpy```, ```matplotlib```, ```tqdm```, ```pandas```, ```scipy```, ```ipython```.

### Additional Dependencies
Some additional dependencies, such as LaTeX fonts and FFmpeg, may be required. We recommend following the installation instructions provided in the ```ssl_simulator``` [README](https://github.com/Swarm-Systems-Lab/ssl_simulator/blob/master/README.md). 


## Usage

We recommend running the Jupyter notebooks in the `src` directory to get an overview of the project's structure and see the code in action.

## Credits

This repository is maintained by [Jesús Bautista Villar](https://sites.google.com/view/jbautista-research). For inquiries or further information, please get in touch with him at <jesbauti20@gmail.com>.
