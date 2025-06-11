# Demo Guide of SkySense-O

Welcome to SkySense-O demo. Here, you only need to follow the instructions below to upload your customed images and specify text to directly obtain pixel-level interpretation results.

## 🚀 Running
After setting up the environment and downloaded the checkpoint, you can execute the following command to enter the demo:
```shell
python demo.py
```
## 🚀 Setting

#### 📝 Configuration Interface
Enter `setting` to access the configuration interface, allowing you to set hyperparameters.
  ```shell
  Example:
    Please input your target texts with ',' split: setting
  ```
  Then you will get following UI. If you need to fill in `T` to confirm , if not, just press Enter without filling in anything.
  ```shell
  Example:
    Custom_image: 
    Custom_text: T
    Custom_save_path: T
  ```  
-----------
#### 📝 Custom Input Text Manner
**Manner 1 :**  || **Customed Image :**  ✅ || **Customed Text :** ❌ ||
- **Open-World Interpretation**: The model executes using node categories from the Sky-SA graph as text input.
  ```shell
  Example:
    Please input your target texts with ',' split: open_world
  ```
**Manner 2 :**  || **Customed Image :**  ✅ || **Customed Text :** ✅ (dataset-level) ||
- **Dataset-Specific Category Output**: Output results based on category names from specific datasets.
  ```shell
  Example:
    Please input your target texts with ',' split: isa_idataset, oem_dataset
  ```
**Manner 3 :**  || **Customed Image :**  ✅ || **Customed Text :** ✅ ||
- **Custom Category Text**: Supports user-defined category names input, separated by commas.
  ```shell
  Example:
    Please input your target texts with ',' split: ground, structure
  ```
-----------
#### 📝 Custom Input Image Path

- Users can modify the input image path when prompted.
  ```shell
  Example:
    Please input your input image path: ./input.jpg
  ```
-----------
#### 📝 Custom Output Image Path
- Users can specify the storage location for the output result image by entering the desired save path.
  ```shell
  Example:
    Please input your save path: ./output.png
  ```



## Contact and Feedback
Thank you for using SkySense-O. If you have any questions or suggestions, please feel free to contact us. (zqcrafts@mail.ustc.edu.cn)

