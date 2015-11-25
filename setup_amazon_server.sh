wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

source .bashrc
conda install -y numpy pandas scikit-learn ipython jupyter matplotlib
sudo yum -y install git tmux
git clone https://github.com/anabranch/mlproject.git
