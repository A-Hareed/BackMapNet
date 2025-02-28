sudo apt update
sudo apt install -y build-essential libssl-dev uuid-dev libgpgme-dev squashfs-tools libseccomp-dev wget pkg-config


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
sudo wget -O /usr/share/keyrings/sylabs-archive-keyring.gpg https://apt.fury.io/sylabs/gpg.
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/sylabs-archive-keyring.gpg] https://apt.fury.io/sylabs/ $distribution main" | sudo tee /etc/apt/sources.list.d/sylabs.list

sudo apt update


singularity --version
