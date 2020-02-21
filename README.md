# Team 4 San Francisco Crime Kaggle Competition Solution


## Team Members

* Scott Duda
* Steve Estes
* George Jiang
* Morgan Kaiser

### Directions

1. Make sure you have the original training and test data from the [competition](https://www.kaggle.com/c/sf-crime) (train.csv and test.csv) saved in the same directory as the notebooks included in this repo. 
2. Run the preprocessing notebook.
3. Using the output files from the preprocessing notebook (p_train.csv and p_test.csv), train each of the individual models using its corresponding notebook. 
4. Using the output files from each model training notebook, run the stacking-combine notebook to create the final submission file using a stacked ensemble created from the predictions of each of the five models. 

### Model Results

**Final Competition Score:  2.19915**
*(evaluated using multi-class logarithmic loss)*

### Additional Information

The code included in this repo was developed as part of the PGH Data Science Meetup Group's 2020 Kaggle Competition. A copy of the presentation that was delivered to present the model results can be found [here](https://docs.google.com/presentation/d/1V-fzkfyjcLuN9bXJlbOhAorDTtMSrGwZ4xY1w1w0geM/edit?usp=sharing).  

An additional set of code was developed for incorporation of external datasets into the original data provided by the competition hosts. This code will be added to the repo in the near future. There were several interesting takeaways from the other team presentations that may inspire future code modifications as well. 
