I have attached some late stage LLM results using the OpenAI Chat interface. 
All modes were default and custom instructions was left empty. No special tool was augmented. (I do not have access to an API currently, but I have a Go subscription. I believe the results will be very similar to calling an API; it is trivial to implement.)

Some ideas include using confidence thresholding to selectively modify words, fine tuning LLMs on transcription files. 

Prompt used:
```
Instruction: "You will be given a transcript of Renaissance era Spanish text generated using an OCR Model. Your job is to clean the transcript preserve meaning and era-related specifics. You must try to preserve the original accents as much as possible. Return the modified transcript only. Do not return or say anything else."
Input: "<--text goes here-->"
```
