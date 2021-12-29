mkdir ../dataset
mkdir ../dataset/esr
wget http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata -P ../dataset/credit/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv -O ../dataset/esr/esr.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z -P ../dataset/isolet/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z -P ../dataset/isolet/
gzip -d ../dataset/isolet/isolet5.data.Z
gzip -d ../dataset/isolet/isolet1+2+3+4.data.Z
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P ../dataset/adult/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -P ../dataset/adult/
sed -i -e '1d' ../dataset/adult/adult.test
mkdir ../dataset/mnist
mkdir ../dataset/fashion