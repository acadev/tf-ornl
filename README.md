# tf-ornl

## Getting Started

  1) Clone this repo on your DGX
  2) Add your `data.pkl` file to the `tf-ornl` folder
  3) Run `./run_tensorflow` in the `tf-ornl` folder, then connect to http://<dgx-ip>:8000
  4) Run the `data-to-tf-record-v2` notebook to create 1024 files in `data_v2`
  5) Run the `tf-train` notebook
