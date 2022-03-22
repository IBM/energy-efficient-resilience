Energy Efficient & Resilient AI
===============================

Training AI models to jointly optimize energy efficiency and resilience.

Required commands
-----------------

* git
* python (3.6 tested)
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

You will see that you command prompt changes. You should be able
to execute the provided related commands. This should be the only
command you need to execute before starting the experiments.

Example
-------

To run an 8-bit model with error injection at the rate of 0.01
python3 zs_main.py resnet18 eval cifar10 -cp checkpoint.pth -ber 0.01

Check and edit if needed config.py for configuration options.

Contacts
--------
Nandhini Chandramoorthy <Nandhini.Chandramoorthy@ibm.com>
Ramon Bertran <rbertra@us.ibm.com>



[^1]:
  Note that the installation of pytorch and other dependencies migth require
  significant amount of space. Plan accordingly to avoid file system related
  issues.
