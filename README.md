EERAI - Energy Efficient & Resilient AI
=======================================

Training AI models to jointly optimize energy efficiency and resilience.

[![Build Status](https://app.travis-ci.com/IBM/energy-efficient-resilience.svg?branch=main)](https://app.travis-ci.com/IBM/energy-efficient-resilience)

![GitHub](https://img.shields.io/github/license/IBM/energy-efficient-resilience.svg)
![GitHub forks](https://img.shields.io/github/forks/IBM/energy-efficient-resilience.svg?style=social)
![GitHub stars](https://img.shields.io/github/stars/IBM/energy-efficient-resilience.svg?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/IBM/energy-efficient-resilience.svg?style=social)
![GitHub license](https://img.shields.io/github/license/IBM/energy-efficient-resilience.svg?style=solcial)

Required commands
-----------------

* git
* python (3.6, 3.7, 3.8, 3.9)
* virtualenv: https://virtualenv.pypa.io/en/stable/

First-time set up
-----------------

We are assuming a bash environment throughout the process. You might
try to use other shell, although some commands might need to be
modified accordingly. Execute the following commands the first
time:

```bash
git clone git@github.com:IBM/energy-efficient-resilience.git INSTALLDIRECTORY
cd INSTALLDIRECTORY
./bootstrap_environment.sh
```

Hopefully, the installation is complete. Otherwise, report the
error to the development team. The commands above did the following:
create a virtual python environment to avoid any dependency issues
(which we found quite often...) and activate it. Then, we have
checked out the repository. Finally, we installed the required
dependencies [^1].

Using EERAI
-----------

Assuming that all the variables above are set in your environment,
you just need to execute the following command to start using the
scripts.

```bash
cd INSTALLDIRECTORY
source activate_eerai
```

You will see that your command prompt changes. You should be able
to execute the provided related commands. This should be the only
command you need to execute before starting the experiments.

Example
-------

Check and edit the `config.py` for configuration options. Main options
of interest are the paths where all the data is going to be stored
(make sure you have enough space to avoid issues) as well as the
default layers to inject errors.

First one need to train the model:

```bash
zs_main.py resnet18 train cifar10 -E 10
```

The previous command will train the model for 10 epochs. Checkpoints
of each epoch will be stored in the configured paths. Then, one can
evaluate the model with:

```bash
zs_main.py resnet18 eval cifar10 -E 10
```

Contacts
--------

- Nandhini Chandramoorthy <Nandhini.Chandramoorthy@ibm.com>
- Ramon Bertran <rbertra@us.ibm.com>

[^1]:
    Note that the installation of pytorch and other dependencies migth require
    significant amount of space. Plan accordingly to avoid file system related
    issues.
