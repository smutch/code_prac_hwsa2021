HWSA 2021 code practical
========================

Slides can be found `here <https://smutch.github.io/hwsa2021-slides/>`_.

A note on the code
------------------

This is not a production or science-ready code. It is only meant to be used as an example code to practice optimisation and play with Python tools. It is modified from a student's coursework project and makes several (well-founded) simplifying assumptions. The code has NOT been validated against known solutions.
*** DO NOT USE THIS FOR RESEARCH! **


Getting set up
--------------

.. highlight: bash

First clone this repo from github::

    git clone git@github.com:smutch/code_prac_hwsa2021.git
    cd code_prac_hwsa2021

Dependencies can be installed using either :ref:`pip <pip-install>` or :ref:`conda(+pip) <conda-install>` depending on your preference.

.. _pip-install:

Pip
^^^

This code has only been tested with Python 3.9. You can check your verison of python using::

    python -V

If you are using an older version then 3.9 then it is strongly recommended that you install it using `pyenv`_ or a similar python version management tool.

Once you have Python 3.9, create a new virtualenv using the tool of your choice and activate it e.g.::

    python -m venv env
    source env/bin/activate

You can then install the `code_prac` in editable mode, and all its dependencies using::

    pip install -e .[dev]

.. _pyenv: https://github.com/pyenv/pyenv

.. _conda-install:

Conda
^^^^^

Setup a new conda environment with all required dependencies and the `code_prac` package installed in editable mode using::

    conda env create -f environment.yml
    conda activate code_prac_hwsa2021
    pip install -e .[dev]
