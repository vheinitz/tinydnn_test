# tinydnn_test
Test C++11 header-only CNN Framefork with ANCA cells. Gained accuracy: 99%.

"Anti-neutrophil cytoplasmic antibodies (ANCAs) are a group of autoantibodies, mainly of the IgG type, 
against antigens in the cytoplasm of neutrophil granulocytes (the most common type of white blood cell) and monocytes. 
They are detected as a blood test in a number of autoimmune disorders, 
but are particularly associated with systemic vasculitis, so called ANCA-associated vasculitides."
    Source: https://en.wikipedia.org/wiki/Anti-neutrophil_cytoplasmic_antibody
    
This esamples evaluated TinyDNN for using with ANCA-paterns. The ells were extracted from grayscale microscope images. 
The cell-images were normalizer, resized to 32x32. CSV-Files were created using 500 cells for training and 500 cells for testing.
There are two classes P-ANCA and C-ANCA in the model.

The csv-files are located in testdata-directory. For testing the code directly, copy the csv-files to c:/tmp/ (or change the paths in code)
