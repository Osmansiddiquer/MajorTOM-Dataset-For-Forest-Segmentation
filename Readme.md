Before working with this, make sure you have git lfs setup
## Credits and Datasets Used:
- The Major-TOM dataset was created by the Major-TOM team. You can find more information about the dataset [here](https://huggingface.co/Major-TOM).
- The FNF4 dataset from the Palsar satellite was also used. You can access the dataset [here](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_FNF4).

## Usage:
Before using the dataset, please explore the dataset usage notebook provided. This will give you a better understanding of how to utilize the dataset effectively.

## Some Notes About Work in Preparing Major-TOM S1RTC and S2-L2A dataset for Self-Supervised Learning:
### Goal: 
The goal is to provide information about the forest ratio of each tile in the Major-TOM datasets and prepare a HF or PyTorch dataset that can be configured to training needs. The aim is to make the data retrieval as quick as possible, while keeping in mind the limited storage space of computers using the dataset. 
### Plan:
-	Understand the Major-TOM datasets and find possible datasets that can be used to get information about the forest percentage. 
-	Use the datasets to write code that prepares/modifies the metadata for each tile to have information about the forest percentage.
-	Execute the code and process the metadata for both datasets (cloud services needed).
    -	The code execution had to be distributed on colab, Kaggle, and multiple local sessions to prepare the metadata, and took weeks between interruptions and errors.
-	Prepare a PyTorch dataset that streams data from the hugging face to provide data for training. A balance between minimizing data loading delays (by using caching, or parallel retrieval of data) and ensuring batch randomization must be reached.
    -	The dataset must provide the images by splitting them into pieces since over 20 bands of 1068x1068 images will make training on available GPUs impossible. So, one strategy that could be used is to have only one batch from a single tile of data. This would allow data to be loaded in parallel while the current batch is being processed but the batch size would be limited to 16 or 32 at max.
    -	The coordinate systems are not the same across the tiles in the dataset. An expert opinion would help but from my reading about and experimenting with the Major-Tom dataset, the Major-TOM grid is independent of any coordinate reference system, and they may only be used as a reference for the provided center latitudes and longitudes of the grid cells/tiles.
    -	The S1-RTC has data available for fewer grid cells than the S2-L2A dataset. As the graph below shows, the data is missing randomly, and no specific trend can be seen. Using the S1-RTC for training would mean that we will lose over 780k grid cells over the globe. For this reason, the dataset will have the option to include S1-RTC data in the tiles that it outputs.
    -	The helper functions provided by my major tom maintainers are in some instances, made for S2 data only and will have to be modified.

## Tips
- Right now, it takes ~5 seconds to load each batch when using Sentinel 1 (S1). This can be reduced by increasing the number of workers and is even lower when not using S1. 
- Set Keep_cache to True to get performance gains at the cost of storage. It will download the data during the first epoch and subsequent epochs will be much faster. 
