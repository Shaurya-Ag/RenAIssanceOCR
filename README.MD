
This is my submission for GSoC HumanAI foundation Proposal Screening Task.
# Run Inference
To run the pipeline, insert your images in `inference\images` folder. Then, in the project root, run
```
python .\inference.py --image_folder .\inference\images\ --result_folder .\inference\results
```
The results will be in the transcripts folder. Please note that no late stage LLM or VLM is applied here. Read [LLM.md](LLM.md) for more details.

# Training
The trainer module is in [Trainer.py](Trainer.py).
It writes to a tensorboard interface: `tensorboard --logdir=runs`.
The CRNN Model is defined in [Model.py](Model.py). 

To see the training details, see [Model Development.ipynb](Model%20Development.ipynb)

My custom annotator tool is available here: [BoxAnnotator](https://github.com/Shaurya-Ag/BoxAnnotator)

The final extraction details are in [extract_data.py](extract_data.py). It may give errors because I did not intend it to be re-used. Some other details for Data Preprocessing are in [Old Extraction Notebook.ipynb](Old%20Extraction%20Notebook.ipynb)

I have changed the CRAFT Model scripts to suit my needs. I don't recommend modifying them.