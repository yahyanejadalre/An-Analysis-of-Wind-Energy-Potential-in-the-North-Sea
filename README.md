# An Analysis of Wind Energy Potential in the North Sea

> In the face of climate change, it is widely agreed that the energy production
> has to rely on more sustainable and renewable forms of harnessing energy.
> Offshore wind turbine parks play a crucial role in increasing the share of green
> energy. This paper explores probabilistic methods of assessing the wind energy
> potential and potential trends in data collected on Helgoland by considering
> wind speeds as a Weibull distributed random variable.  Further, for forecasting,
> the monthly expected wind power density is extrapolated by a Gaussian process
> regression model.

This work was done as part of Data Literacy course at University of Tübingen.
This is the repository containing all the relevant figures, source and tex
files.

## Content

1. `notes`: A directory contains multiple jupyter notebooks used in the analysis.
This is where all figures are created.
2. `util`: A directory contains python scripts used in the analysis.
3. `Makefile`: Used for compiling the tex file to a pdf, deleting the data folder and
other tasks. See below for a more in-depth description.

## Installation & Usage

First, setup the a new Python environment using conda. Simply run `conda env
create -f environment.yml` to setup a new enivornment called `dataliteracy`.
Then, run `conda activate` and `pip install -r requirements.txt` to install all
needed dependencies. For reproducing all the used plots, run the notesbook in
`notes/`. Finally, use `make pdf` to build the paper's PDF. See below for
other `make` commands:

#### Delete the downloaded data

```bash
make clean
```

#### Create the article pdf

```bash
make pdf
```

#### Delete the article pdf

```bash
make clean-pdf
```
