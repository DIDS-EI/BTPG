# BTPGym

 Platform and Benchmark for Behavior Tree Planning in Everyday Service Robots. Based on [VirtualHome](http://virtual-home.org/) v2.3.0


# Installation

Create a conda environment.
```shell
conda create --name BTPGym python=3.9
conda activate BTPGym
```

Install BTPGym.
```shell
cd BTPGym
pip install -e .
```

Download the VirtualHome executable for your platform (Only Windows is tested now):

| Operating System | Download Link                                                                      |
|:-----------------|:-----------------------------------------------------------------------------------|
| Linux            | [Download](http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip)   |
| MacOS            | [Download](http://virtual-home.org/release/simulator/v2.0/v2.3.0/macos_exec.zip)   |
| Windows | [Download](http://virtual-home.org/release/simulator/v2.0/v2.3.0/windows_exec.zip) |  


# Usages

1. Download the simulator ([windows version](http://virtual-home.org/release/simulator/v2.0/v2.3.0/windows_exec.zip))
2. Unzip all files in windows_exec.v2.2.4 and move them to simulators/virtualhome/windows.
3. Run the test/watch_tv.py and see the simulation result.
```python
python test_exp/main.py
```

