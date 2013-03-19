README for StatML assignments
Authors: Asbjoern Thelger, Joachim Vig, Andreas Bock

We use python and the following packages have to be installed to run our code:

scipy
numpy
pylab
libsvm
PIL (Python Image Library)
PyBRAIN

INSTALLATION:

On Arch Linux, this can be done be executing the following command:

sudo pacman -S python-scipy python-numpy python-matplotlib python-imaging
sudo yaourt -S pybrain

On Ubuntu:

sudo apt-get install python-numpy python-scipy python-matplotlib python-imaging

and then

git clone git://github.com/pybrain/pybrain.git pybrain
python setup.py install
