# ibmds-mnbvc


## Convert and visualize text bounding boxes

 
 In this example we will use the output of the converted document and create an image of elements detected on each page.
 
 
### System dependencies
 This script is converting a PDF document with Deep Search and exports the figures into PNG files.
 
 The PDF to image conversion relies on the `pdftoppm` executable of the Poppler library (GPL license)
 https://poppler.freedesktop.org/
 The Poppler library can be installed from the most common packaging systems, for example
 - On macOS, `brew install poppler`
 - On Debian (and Ubuntu), `apt-get install poppler-utils`
 - On RHEL, `yum install poppler-utils`
 

### Notebooks parameters
 
 The following block defines the parameters used to execute the notebook
 
 - `INPUT_FILE`: the input PDF to converted and analyzed
 - `SHOW_PDF_IMAGE`: if enabled, the background will contain the rendered PDF page
 - `SHOW_CLUSTER_BOXES`: if enabled, the cluster boxes will be visualized
 - `SHOW_TEXT_CELLS_BOXES`: if enabled, the PDF raw text cells will be visualized
 
