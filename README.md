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

## Credits
This repository is maintained by [Jesús Bautista Villar](https://sites.google.com/view/jbautista-research). For inquiries, suggestions, or further information, feel free to contact him at <jesbauti20@gmail.com>.
