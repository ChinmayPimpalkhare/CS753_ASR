## CS753 - Hacker Role; Team Audio_Linguists

## Paper Title - AlignCTC

"ALIGN WITH PURPOSE: OPTIMIZE DESIRED PROPERTIES IN CTC MODELS WITH A GENERAL PLUG-AND PLAY FRAMEWORK" 

### AWP-AlignWithPurpose

AWP is a versatile PLUG-AND-PLAY framework designed to optimize specific properties in Connectionist Temporal Classification (CTC) models.

In this code, we have included an implementation of the f_low_latency function, which serves as the shift function for optimizing the low latency property of a CTC model. You can further extend this framework by implementing additional shift functions tailored to other properties.

### Getting Started
To get started with AWP, follow these steps:
1. Import the AlignWithPurpose class into your CTC model project.
2. Implement your custom shift functions, such as f_low_latency, to align the model with your desired property.
3. Define a weight for the AWP loss, which will determine the relative importance of optimizing the specified property compared to the CTC loss.

### Running Example
An example of how to use AWP with your CTC model:
```python
# Import the necessary modules
from awp import AlignWithPurpose

# Create an instance of the AlignWithPurpose class with a N number of paths to sample. 
loss_AWP_fn = AlignWithPurpose(num_of_paths=5, device=device)

# Calculate the AWP loss by providing the model output, shift function, and model prediction length
loss_AWP = loss_AWP_fn(output_train, shift_function="f_low_latency", model_pred_length=model_pred_time_length)

# Scale the AWP loss by the defined weight
loss_AWP = loss_AWP * loss_AWP_weight

# Calculate the total loss as a combination of CTC loss and AWP loss
loss = loss_CTC + loss_AWP
```


