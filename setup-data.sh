curl https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_images.zip -o mnist_images.zip
curl https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_labels.csv -o mnist_images.csv
unzip mnist_images.zip

# Feel free to adequately change this
INFERENCE_DATA_PATTERN="0-0[0-2][0-9].png"

mkdir data_for_inference
cd mnist_images 
mv `ls | grep "$INFERENCE_DATA_PATTERN"` ../data_for_inference
cd ..

grep -v "$INFERENCE_DATA_PATTERN" mnist_images.csv > mnist_images.csv.temp
rm mnist_images.csv
mv mnist_images.csv.temp mnist_images.csv

mkdir data_for_inference/more_data
cp 4.jpg data_for_inference/more_data